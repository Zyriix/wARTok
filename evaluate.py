"""Pure evaluation: no optimizers, training steps, or EMA updates."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import ImageFolderDataset
from omegaconf import OmegaConf
import torch.nn.functional as F
from models import Encoder, Decoder, ARModel
from train import (seed_everything, patchify, img_unint8_to_norm, DiTiScheduler,
                   reconstruction, sampling, euler_recon_loop, euler_sample_loop,
                   img_norm_to_uint8, unpatchify)


def state_dict(state):
    return {k.removeprefix('_orig_mod.'): v for k, v in state.items()}


def load_models(config, checkpoint, device='cuda', ema=True, need_ar=True):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    enc, dec = Encoder(config), Decoder(config)
    enc.load_state_dict(state_dict(state['enc']), strict=True)
    dec.load_state_dict(state_dict(state['dec']), strict=True)
    ar = None
    if need_ar:
        ar = ARModel(config)
        key = 'ar_model_ema' if ema else 'ar_model'
        if state.get(key) is None:
            raise ValueError(f'Checkpoint has no {key}; select a matching AR checkpoint')
        ar.load_state_dict(state_dict(state[key]), strict=True)
    for model in (enc, dec, ar):
        if model is not None:
            model.to(device).eval().requires_grad_(False)
    return enc, dec, ar


def scheduler(config):
    if config.get('time_stages') and config.get('k_per_stage'):
        return DiTiScheduler(1000, config.z_len, config.time_stages,
                             config.k_per_stage, config.reverse_t)
    return None


def labels_for(config, batch_size, device):
    if config.use_label:
        raise ValueError('The released CIFAR-10 protocols are unconditional')
    return F.one_hot(torch.full((batch_size,), config.num_classes-1,
                               device=device, dtype=torch.long), config.num_classes).float()


@torch.no_grad()
def decode_batch(config, enc, dec, ar, x, labels, mode, diti=None):
    """x is a patchified batch; its shape also defines unconditional generation."""
    t = torch.ones(x.shape[0], device=x.device) if config.use_noise_query else None
    # wARTok draws reconstruction noise before the generation branch, even for AR-only eval.
    noise_recon = torch.randn_like(x) if config.use_noise_query else None
    if mode == 'reconstruct':
        if config.use_noise_query and config.diffusion_decoder:
            result, _ = euler_recon_loop(enc, dec, x, labels, config=config, diti_scheduler=diti)
        else:
            result, _ = reconstruction(enc, dec, x, labels, query_tokens=noise_recon, t=t)
            if config.l2_v_target:
                result = noise_recon - result
    elif mode == 'generate':
        kwargs = dict(bz=x.shape[0], class_label=labels, temperature=config.temperature,
                      topK=config.topK, topP=config.topP, cfg=config.cfg,
                      cfg_schedule=config.cfg_schedule, cfg_power=config.cfg_power,
                      cache_kv=config.cache_kv, config=config)
        if config.use_noise_query and config.diffusion_decoder:
            result = euler_sample_loop(enc, dec, ar, **kwargs, diti_scheduler=diti)
        else:
            noise = torch.randn_like(x) if config.use_noise_query else None
            result = sampling(enc, dec, ar, **kwargs, t=t, query_tokens=noise)
            if config.l2_v_target:
                result = noise - result
    else:
        raise ValueError(mode)
    return img_norm_to_uint8(unpatchify(result, config.image_size, config.patch_size))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=['generate', 'reconstruct'], default='generate')
    parser.add_argument('--data', help='Required for reconstruction; use the CIFAR-10 training split')
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64, help='Per process batch size (original evaluation: 64)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--legacy', action=argparse.BooleanOptionalAction, default=True,
                        help='Use the same seed on every rank (wARTok behavior); --no-legacy uses seed + rank')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--fid-reference')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        parser.error('sample count and batch size must be positive')
    world = int(os.environ.get('WORLD_SIZE', 1)); rank = int(os.environ.get('RANK', 0))
    if world > 1:
        if args.device != 'cuda': parser.error('distributed evaluation requires CUDA')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        # Ranks exchange files, not CUDA tensors; CPU barriers avoid requiring GPU P2P.
        torch.distributed.init_process_group('gloo')
    c = OmegaConf.load(args.config)
    torch.backends.cuda.matmul.allow_tf32 = bool(c.use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(c.use_tf32)
    seed_everything(args.seed)
    enc, dec, ar = load_models(c, args.checkpoint, args.device, not args.no_ema,
                               args.mode == 'generate')
    diti = scheduler(c)
    # Strided partition, without DistributedSampler padding: exactly N images globally.
    indices = list(range(rank, args.num_samples, world))
    loader = None
    if args.mode == 'reconstruct':
        if not args.data: parser.error('--data is required for reconstruction')
        dataset = ImageFolderDataset(path=args.data, resolution=c.image_size,
                                     use_label=False, xflip=False, crop_type=c.eval_crop_type)
        if args.num_samples > len(dataset): parser.error('num-samples exceeds dataset size')
        loader = iter(DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False))
    # A fixed evaluation seed is explicit; historical training-loop RNG was not checkpointed.
    seed_everything(args.seed + (0 if args.legacy else rank))
    samples = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(indices), args.batch_size), desc=args.mode, disable=rank != 0):
            n = min(args.batch_size, len(indices)-start)
            if loader is None:
                x = torch.empty(n, c.x_len, c.patch_dim, device=args.device)
            else:
                images, _ = next(loader)
                x = patchify(img_unint8_to_norm(images.to(args.device)), c.patch_size)
            labels = labels_for(c, n, args.device)
            images = decode_batch(c, enc, dec, ar, x, labels, args.mode, diti)
            samples.append(images.permute(0,2,3,1).cpu().numpy())
    arr = np.concatenate(samples) if samples else np.empty((0,c.image_size,c.image_size,3),dtype=np.uint8)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        np.savez(out/f'rank-{rank}.npz', images=arr, indices=np.asarray(indices))
        torch.distributed.barrier()
    if rank == 0:
        if world > 1:
            arr = np.empty((args.num_samples,c.image_size,c.image_size,3),dtype=np.uint8)
            for r in range(world):
                with np.load(out/f'rank-{r}.npz') as shard: arr[shard['indices']]=shard['images']
        assert len(arr) == args.num_samples
        path = out/'samples.npz'; np.savez_compressed(path, arr)
        save_image(torch.from_numpy(arr[:64]).permute(0,3,1,2).float()/255, out/'preview.png',nrow=8)
        report = dict(mode=args.mode, num_samples=len(arr),
                      unique_images=len({hashlib.sha256(im.tobytes()).digest() for im in arr}), seed=args.seed,
                      world_size=world, batch_size=args.batch_size, legacy=args.legacy, rank_seed_mode='shared' if args.legacy else 'independent', config=args.config,
                      checkpoint=args.checkpoint, ar_ema=(not args.no_ema) if args.mode == 'generate' else None,
                      decoder='euler25' if c.use_noise_query and c.diffusion_decoder else 'single_step',
                      precision='32-true', fid=None, torch_version=torch.__version__,
                      config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
        if args.fid_reference:
            report['reference_sha256']=hashlib.sha256(Path(args.fid_reference).read_bytes()).hexdigest()
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('calc_fid.py')),
                '--ref_batch', args.fid_reference, '--sample_batch', str(path), '--batch_size','64'],
                text=True, check=True, stdout=subprocess.PIPE)
            print(result.stdout)
            report['fid']=float(next(line.split(':',1)[1] for line in result.stdout.splitlines() if line.startswith('FID_RESULT:')))
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report, indent=2))
    if world > 1:
        torch.distributed.barrier(); torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
