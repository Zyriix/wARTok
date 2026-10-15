import itertools
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple
os.environ.setdefault("WANDB_MODE", "disabled")
import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils
from einops import rearrange
from lightning import Fabric
from wandb.integration.lightning.fabric import WandbLogger
from dataset import ImageFolderDataset
from gan_loss import GANLoss
from lpips import LPIPS
from models import ARModel, Encoder, Decoder, Linear
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import copy
import torch.distributed as dist
from glob import glob
try:
    from thop import profile
    from thop.vision.basic_hooks import count_linear
except ImportError:
    profile = None
    count_linear = None


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Phase utils
class Target(NamedTuple):
    DO_AE: bool = False
    DO_L2: bool = False
    DO_L1: bool = False
    DO_LPIPS: bool = False
    DO_GAN_G: bool = False
    DO_GAN_D: bool = False
    DO_PRIOR_ENC: bool = False
    DO_PRIOR_AR: bool = False

class Phase(NamedTuple):
    num_steps: int
    targets: List[Target]       
    internal_steps: List[int]   

# Parse phases such as "1000:DO_L2-DO_PRIOR_AR:1 7000:DO_PRIOR_AR:1".
def parse_phases(phases_str):
    phases = [] # list of Phase objects
    for phase_str in phases_str.split(' '):
        num_steps, targets_str, internal_steps_str = phase_str.split(':')
        num_steps = int(num_steps)
        targets = [Target(**{k: True for obj in target_str.split(',') for k in obj.split('-') }) for target_str in targets_str.split(',')]
        internal_steps = [int(step) for step in internal_steps_str.split(',')]
        phases.append(Phase(num_steps, targets, internal_steps))
    return phases

def get_phase(global_step, phases, phase_step_accum, config):
    target=None
    for phase_idx, phase_step in enumerate(phase_step_accum):
        if global_step <= phase_step:
            internel_step = (global_step - phase_step_accum[phase_idx-1]) if phase_idx > 0 else global_step
            internel_accumulate = list(itertools.accumulate(phases[phase_idx].internal_steps))
            internel_step = internel_step % internel_accumulate[-1]

            for inner_idx in range(len(internel_accumulate)):
                if internel_step < internel_accumulate[inner_idx]:
                    target = phases[phase_idx].targets[inner_idx]
                    break
            if target is not None:
                break

    DO_AE = any([target.DO_L2, target.DO_L1, target.DO_LPIPS, target.DO_GAN_G])

    # assert inconsistency configuration
    assert not (DO_AE and  not config.train_ae), f"train_ae={config.train_ae} is inconsistent with DO_AE={DO_AE} in phases"
    assert not (target.DO_PRIOR_AR and  not config.train_ar), f"train_ar={config.train_ar} is inconsistent with DO_PRIOR_AR={target.DO_PRIOR_AR} in phases"
    assert not ((target.DO_GAN_G or target.DO_GAN_D) and  not config.use_gan_loss), f"use_gan_loss={config.use_gan_loss} is inconsistent with DO_GAN_G or DO_GAN_D={(target.DO_GAN_G or target.DO_GAN_D)} in phases"
    assert not (target.DO_PRIOR_ENC and  not config.use_prior_loss), f"use_prior_loss={config.use_prior_loss} is inconsistent with DO_PRIOR_ENC={target.DO_PRIOR_ENC} in phases"
    assert not (target.DO_LPIPS and  not config.use_lpips_loss), f"use_lpips_loss={config.use_lpips_loss} is inconsistent with DO_LPIPS={target.DO_LPIPS} in phases"

    # update target with global_step
    target = Target(DO_L1=target.DO_L1, 
                    DO_L2=target.DO_L2, 
                    DO_LPIPS=target.DO_LPIPS, 
                    DO_GAN_G=target.DO_GAN_G and (global_step >= config.gan_start),
                    DO_GAN_D=target.DO_GAN_D and (global_step >= config.gan_start),
                    DO_PRIOR_ENC=target.DO_PRIOR_ENC and (global_step >= config.prior_start),
                    DO_PRIOR_AR=target.DO_PRIOR_AR,
                    DO_AE=DO_AE)
    return phase_idx, inner_idx, target, internel_step

class InfiniteIterator(Iterator):
    def __init__(self, iterable: Iterable):
        self.iterable = iterable
        self._it = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.iterable)
            return next(self._it)

# Image Processing Utils
def patchify(x, patch_size):
    x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    return x

def unpatchify(x, image_size, patch_size):
    x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size, w=image_size//patch_size)
    return x

def img_unint8_to_norm(x):
    return x.float() / 127.5 - 1.0

def img_denormalize(x):
    return x.clamp(-1,1) * 0.5 + 0.5

def img_norm_to_uint8(x):
    return torch.clamp(127.5 * x + 128.0, 0, 255).byte()

# -----------------------------
# Tail-dropout + noise-query utils
# -----------------------------

def _get_decoder_context_type(dec: nn.Module) -> str:
    # `dec` is Decoder, `dec.dec` is Transformer
    ctx = getattr(getattr(dec, "dec", None), "context_type", None)
    return ctx if ctx is not None else "none"

def _get_decoder_query_len(dec: nn.Module) -> int:
    return int(getattr(getattr(dec, "dec", None), "max_seq_len", 0))

def _t_to_k(t: torch.Tensor, K: int, t2k: float = 1.0) -> torch.Tensor:
    """
    Map t in [0,1] to discrete k in [0, K-1].
    """
    if K <= 1:
        return torch.zeros_like(t, dtype=torch.long)
    t_tmp = (t2k * t).clamp(0.0, 1.0)
    k = torch.floor(t_tmp * K).to(torch.long).clamp(0, K - 1)
    return k

class DiTiScheduler:
    """
    Piecewise linear time scheduler from Selftok, adapted for wARTok.
    """
    def __init__(
        self,
        n_timesteps: int,
        K: int,
        stages: str | List[int],
        k_per_stage: str | List[int],
        reverse_t: bool = False,
    ):
        self.n_timesteps = n_timesteps
        self.K = K
        self.reverse_t = reverse_t

        if isinstance(stages, str):
            stages = [int(s.strip()) for s in stages.split(',')]
        if isinstance(k_per_stage, str):
            k_per_stage = [int(k.strip()) for k in k_per_stage.split(',')]
        self.stages = stages
        
        # Scaling logic: if sum(k_per_stage) != K, scale it
        total_k = sum(k_per_stage)
        if total_k != K:
            print(f"[DiTiScheduler] Warning: sum(k_per_stage)={total_k} does not match K={K}. Scaling k_per_stage...")
            scale_factor = K / total_k
            scaled_k = [int(k * scale_factor) for k in k_per_stage]
            # Fix rounding errors to ensure sum is exactly K
            diff = K - sum(scaled_k)
            if diff != 0:
                scaled_k[-1] += diff
            k_per_stage = scaled_k
            print(f"[DiTiScheduler] Scaled k_per_stage: {k_per_stage}")
            
        self.k_per_stage = k_per_stage
        
        self.t_to_idx = torch.zeros(n_timesteps).long()
        
        # Precompute lookup table
        current_stage = 0
        sum_indices = 0
        # stages should be cumulative, e.g., 200, 400, 600...
        
        for t in range(n_timesteps):
            # Check if we moved to next stage
            if current_stage < len(self.stages) and t >= self.stages[current_stage]:
                sum_indices += self.k_per_stage[current_stage]
                current_stage += 1
            
            if current_stage >= len(self.stages):
                 idx = self.K - 1
            else:
                prev_stage_t = self.stages[current_stage - 1] if current_stage > 0 else 0
                current_stage_t = self.stages[current_stage]
                
                current_steps = float(current_stage_t - prev_stage_t)
                current_k = float(self.k_per_stage[current_stage])
                
                t_adj = t - prev_stage_t
                idx = int(float(t_adj) / current_steps * current_k + sum_indices)
            
            self.t_to_idx[t] = idx

    def to_indices(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert t in [0, 1] to indices [0, K-1] using the lookup table.
        """
        if self.reverse_t:
            t = 1.0-t
        device = t.device
        # Map [0, 1] to [0, n_timesteps]
        t_idx = (t * self.n_timesteps).long().clamp(0, self.n_timesteps - 1)
        # Move the lookup table to the input device.
        return self.t_to_idx.to(device)[t_idx].clamp(0, self.K - 1)


def build_taildrop_attn_mask(
    *,
    t: torch.Tensor,
    context_len: int,
    query_len: int,
    context_type: str,
    t2k: float = 1.0,
    device=None,
    diti_scheduler: DiTiScheduler = None,
) -> torch.Tensor | None:
    """
    Build a boolean attention mask (True for keep, False for masked).

    - cross_attn: returns (B, 1, query_len, context_len)
    - concat:     returns (B, 1, total_len, total_len) where total_len = query_len + context_len
    """
    device = device or t.device
    B = t.shape[0]
    if context_len <= 0:
        return None

    if diti_scheduler is not None:
        k = diti_scheduler.to_indices(t) # (B,)
    else:
        k = _t_to_k(t, context_len, t2k=t2k)  # (B,)
        
    keep = (torch.arange(context_len, device=device)[None, :] <= k[:, None])  # (B, context_len)


    if context_type == "cross_attn":
        # (B, 1, query_len, context_len)
        return keep[:, None, :].unsqueeze(1).expand(B, 1, query_len, context_len)

    if context_type == "concat":
        total_len = query_len + context_len
        keep_total = torch.cat(
            [torch.ones((B, query_len), device=device, dtype=torch.bool), keep],
            dim=1,
        )  # (B, total_len)
        # (B, 1, total_len, total_len)
        return keep_total[:, None, :].unsqueeze(1).expand(B, 1, total_len, total_len)

    return None

def make_noise_image_patch_query(x: torch.Tensor, t: torch.Tensor, query_len: int) -> torch.Tensor:
    """
    Replace dummy query with a noised version of the (clean) image patches x.
    x: (B, L, D)
    t: (B,)
    Returns noised patches, noise, and velocity, each of shape (B, L, D).
    """
    B, L, D = x.shape
    noise = torch.randn_like(x)
    t = t.view(B, 1, 1).to(x.dtype)
    x_noised = (1.0 - t) * x + t * noise
    return x_noised, noise, noise-x

# Seed utils
def seed_everything(seed):
    # manually call these to ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # fabric seed everything
    Fabric.seed_everything(seed, workers=True)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_worker_init_fn(base_seed: int):
    base_seed = int(base_seed) % (2**32)

    def _init(worker_id: int):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        worker_seed = (base_seed + worker_id + 1000 * rank) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


# FID Utils                    
def adm_fid_evaluator(sample_cached_path, gt_cache_path, config, fabric):
    if not os.path.exists(gt_cache_path):
        raise FileNotFoundError(f"Ground-truth cache not found: {gt_cache_path}")
    if not os.path.exists(sample_cached_path):
        raise FileNotFoundError(f"Sample cache not found: {sample_cached_path}")

    fid_script = str(Path(__file__).with_name('calc_fid.py'))
    env = os.environ.copy()
    cmd = [sys.executable,fid_script,"--ref_batch",gt_cache_path,"--sample_batch",sample_cached_path,"--batch_size",str(config.eval_batch_size),]
    fabric.print(f"Running FID evaluation via {fid_script}...")

    FID = None
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        if line:
            fabric.print(line, flush=True)
        if line.startswith("FID_RESULT:"):
            try:
                FID = float(line.split("FID_RESULT:")[1].strip())
            except ValueError:
                pass
    retcode = process.wait()
    if retcode != 0 or FID is None or not math.isfinite(FID):
        raise RuntimeError(f"FID evaluation failed: exit={retcode}, result={FID}")

    return FID


# Training Utils
@torch.no_grad()
@torch._dynamo.disable
def ema_update(model, ema_model, ema_rate):
    if model is None or ema_model is None:
        return
    
    for p,ema_p in zip(model.parameters(),ema_model.parameters()):
            ema_p.copy_(p.detach().lerp(ema_p, ema_rate))
    


@torch.no_grad()
@torch._dynamo.disable
def sampling(enc, dec, ar_model,  bz, class_label=None, temperature=1.0, topK=0, topP=0., cfg=0.0, cfg_schedule=None, cfg_power=None, cache_kv=False, config=None,t=None,query_tokens=None, **params):
    token_ids = ar_model.sampling(bz, class_label, temperature, topK, topP, cfg, cfg_schedule, cfg_power, cache_kv)
    quant = enc.quantizer.get_codes_w_indices(token_ids)
    decoded = dec(quant, class_label, attn_mask=None, query_tokens=query_tokens,t=t)
    return decoded

@torch.no_grad()
@torch._dynamo.disable
def reconstruction(enc, dec, x, labels, return_idx: bool = False, query_tokens=None,t=None):
    quant, idx, _, _ = enc(x,labels, training=False)
    decoded = dec(quant, labels, attn_mask=None, query_tokens=query_tokens, t=t)
    return decoded, idx

@torch.no_grad()
@torch._dynamo.disable
def euler_sample_loop(enc, dec, ar_model,  bz, class_label=None, temperature=1.0, topK=None, topP=None, cfg=1.0, cfg_schedule=None, cfg_power=None, cache_kv=False, config=None, diti_scheduler=None):
    """
    Euler solver for Flow Matching with dynamic token masking (DiTi).
    """
    if config is None:
        raise ValueError("config is required for euler_sample_loop")
    # Derive device / shape from config to avoid relying on outer-scope globals.
    device = next(dec.parameters()).device
    patch_dim = config.patch_dim
    token_ids = ar_model.sampling(bz, class_label, temperature, topK, topP, cfg, cfg_schedule, cfg_power, cache_kv)
    quant = enc.quantizer.get_codes_w_indices(token_ids)
    query_len = config.x_len

    num_steps = 25
    x = torch.randn((bz, query_len, patch_dim), device=device)
    context_len = quant.shape[1]
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_curr = timesteps[i]
        t_tensor = torch.full((bz,), t_curr, device=device)
        
        attn_bias = None
        if config.use_tail_dropout and config.tied_timestep:
            if diti_scheduler is not None:
                attn_bias = build_taildrop_attn_mask(
                    t=t_tensor, context_len=context_len, query_len=query_len,
                    context_type=config.context_type, device=device, diti_scheduler=diti_scheduler,
                )
            else:
                attn_bias = build_taildrop_attn_mask(
                    t=t_tensor, context_len=context_len, query_len=query_len,
                    context_type=config.context_type, t2k=config.tail_dropout_t2k, device=device,
                )
        decoded = dec(quant, class_label, attn_mask=attn_bias, query_tokens=x, t=t_tensor)
        if config.l2_v_target:
            v_pred = decoded
            x = x - v_pred * dt
        else:
            x0_pred = decoded
            t_frac = dt/t_tensor
            t_frac = t_frac[:,None,None]
            x = x * (1-t_frac) +  x0_pred * t_frac
    return x

@torch.no_grad()
@torch._dynamo.disable
def euler_recon_loop(enc, dec, x, class_label=None, config=None, diti_scheduler=None):
    """
    Euler solver for Flow Matching with dynamic token masking (DiTi).
    """
    if config is None:
        raise ValueError("config is required for euler_recon_loop")
    # Derive device / shape from config to avoid relying on outer-scope globals.
    device = next(dec.parameters()).device
    B = x.shape[0]

    quant, idx, _, _ = enc(x,class_label, training=False)
    context_len = quant.shape[1]
    query_len = x.shape[1]

    num_steps = 25
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps
    xt = torch.randn_like(x)
    for i in range(num_steps):
        t_tensor = torch.full((B,), timesteps[i], device=device)
        attn_bias = None
        if config.use_tail_dropout and config.tied_timestep:
            if diti_scheduler is not None:
                attn_bias = build_taildrop_attn_mask(
                    t=t_tensor, context_len=context_len, query_len=query_len,
                    context_type=config.context_type, device=device, diti_scheduler=diti_scheduler,
                )
            else:
                attn_bias = build_taildrop_attn_mask(
                    t=t_tensor, context_len=context_len, query_len=query_len,
                    context_type=config.context_type, t2k=config.tail_dropout_t2k, device=device,
                )
        decoded = dec(quant, class_label, attn_mask=attn_bias, query_tokens=xt, t=t_tensor)
        if config.l2_v_target:
            v_pred = decoded
            xt = xt - v_pred * dt
        else:
            x0_pred = decoded
            t_frac = dt/t_tensor
            t_frac = t_frac[:,None,None]
            xt = xt * (1-t_frac) +  x0_pred * t_frac
    return xt, idx

def toogle_train_eval(model, train=True):
    if model is None: return
    if train:
        model.train()
    else:
        model.eval()

def toogle_require_grad(model, grads=True):
    if model is None:
        return
    model.requires_grad_(grads)

def zero_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.nan_to_num_(nan=0.0, posinf=1e5, neginf=-1e5)

# Model Prams and GFLOPS Utils
def print_model_info(fabric,model, inputs, name="Model"):
    if model is None: return
    if profile is None:
        fabric.print(f"[{name}] thop not installed, skipping GFLOPS calculation.")
        params = sum(p.numel() for p in model.parameters())
        fabric.print(f"[{name}] Params: {params/1e6:.2f}M")
        return

    # Move inputs to device
    device = next(model.parameters()).device
    inputs = tuple(i.to(device) if isinstance(i, torch.Tensor) else i for i in inputs)

    try:
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        custom_ops = {}
        if count_linear is not None:
            custom_ops[Linear] = count_linear

        macs, params = profile(model_copy, inputs=inputs, custom_ops=custom_ops, verbose=False)
        gflops = macs * 2 / 1e9
        fabric.print(f"[{name}] Params: {params/1e6:.2f}M, GFLOPS: {gflops:.2f}")
        del model_copy
    except Exception as e:
        fabric.print(f"[{name}] Failed to calculate GFLOPS: {e}")
        params = sum(p.numel() for p in model.parameters())
        fabric.print(f"[{name}] Params: {params/1e6:.2f}M")
        
def print_model_stats(fabric,config, enc, dec, ar_model):
    x_len = int(config.x_len)
    patch_dim = int(config.patch_dim)
    num_classes = int(config.num_classes)
    z_len = int(config.z_len)
    z_dim = int(config.z_dim)
    codebook_size = int(config.codebook_size)
    
    x_dummy = torch.randn(1, x_len, patch_dim)
    label_dummy = torch.zeros(1, num_classes)
    
    print_model_info(fabric,enc, (x_dummy, label_dummy), "Encoder")
    
    z_dummy = torch.randn(1, z_len, z_dim)
    # If Decoder uses noise query, it requires query_tokens (and optionally t) at forward time.
    query_type = getattr(getattr(dec, "dec", None), "query_type", "learnable")
    if query_type == "noise":
        query_len = int(getattr(config, "x_len", z_len))
        query_dummy = torch.randn(1, query_len, patch_dim)
        t_dummy = torch.zeros(1)
        print_model_info(fabric, dec, (z_dummy, label_dummy, None, query_dummy, t_dummy), "Decoder")
    else:
        print_model_info(fabric,dec, (z_dummy, label_dummy), "Decoder")
    
    if ar_model is not None:
        idx_dummy = torch.randint(0, codebook_size, (1, z_len))
        print_model_info(fabric,ar_model, (idx_dummy, label_dummy), "ARModel")


def train(config):
    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Parse devices
    devices_value = config.devices
    if isinstance(devices_value, str) and "," in devices_value:
        devices_for_fabric = [int(d.strip()) for d in devices_value.split(',')]
        num_devices = len(devices_for_fabric)
    else:
        try:
            num_devices = int(devices_value)
            devices_for_fabric = num_devices
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for devices: {devices_value}. Expected an integer or comma-separated string like '0,1,2'.")

    if config.batch_size % num_devices or config.batch_size < num_devices:
        raise ValueError("Global batch_size must be divisible by the number of devices")
    if config.eval_data_dir is not None and not Path(config.eval_fid_ref_path).is_file():
        raise FileNotFoundError("Prepare eval_fid_ref_path before training, or set eval_data_dir=null to disable periodic evaluation")

    # prepare seed and save dir
    if config.seed is not None:
        seed = int(config.seed) if config.seed is not None else 0
    seed_everything(seed)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)
    worker_init = make_worker_init_fn(seed)

    if "EXPERIMENT_SAVE_DIR" in os.environ:
        save_dir = os.environ["EXPERIMENT_SAVE_DIR"]
    else:
        experiment_index = len(glob(f"{config.save_dir}/*"))
        save_dir = config.save_dir+f"/{experiment_index:03d}"
        os.environ["EXPERIMENT_SAVE_DIR"] = save_dir
    logger = WandbLogger(project=config.wandb_project, save_dir=save_dir, name=config.wandb_name)

    # initialize fabric
    fabric = Fabric(accelerator="gpu", devices=devices_for_fabric, precision=config.precision, loggers=logger)
    fabric.launch()
    fabric.print(config)
    fabric.print(f"Global seed set to {config.seed}")

    # Initialize DiTi Scheduler if config provided
    diti_scheduler = None
    if getattr(config, "time_stages", None) and getattr(config, "k_per_stage", None):
        fabric.print("Initializing DiTi Scheduler for piecewise linear time-to-k mapping...")
        diti_scheduler = DiTiScheduler(
            n_timesteps=1000, 
            K=int(config.z_len),
            stages=config.time_stages,
            k_per_stage=config.k_per_stage,
            reverse_t=config.reverse_t
        )

    # Models and optimizers
    enc = Encoder(config).to(fabric.device) # always initialize encoder
    dec = Decoder(config).to(fabric.device) 
    if not config.train_ae:
        enc = enc.requires_grad_(False)
        dec = dec.requires_grad_(False)
    ar_model = ARModel(config).to(fabric.device) if config.train_ar else None 
    gan_loss = GANLoss(**config.GANLoss).to(fabric.device) if config.use_gan_loss else None
    perceptual_loss = LPIPS().to(fabric.device).eval().requires_grad_(False) if config.use_lpips_loss else None


    # Load Ckpt
    if config.ae_ckpt_path != "":
        fabric.print(f'resume ae from ckpt {config.ae_ckpt_path}')

        ckpt_states = fabric.load(config.ae_ckpt_path)
        if enc is not None and ckpt_states.get('enc', None) is not None:
            enc.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ckpt_states['enc'].items() })
            fabric.print("loaded enc")
        if dec is not None and ckpt_states.get('dec', None) is not None:
            dec.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ckpt_states['dec'].items() })
            fabric.print("loaded dec")

        if gan_loss is not None and ckpt_states.get('gan_loss', None) is not None:
            gan_loss.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ckpt_states['gan_loss'].items() })

    if config.ar_ckpt_path!="":
        fabric.print(f'resume ar from ckpt {config.ar_ckpt_path}')
        ar_ckpt_states = fabric.load(config.ar_ckpt_path)
        if config.load_ar_model and ar_model is not None and ar_ckpt_states.get('ar_model', None) is not None:
            ar_model.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ar_ckpt_states['ar_model'].items() } )
            fabric.print("loaded ar")

    # for ema
    ae_ema_rate = math.pow(0.5,1/config.ae_ema_halflife) if config.ae_ema_halflife>0 else 0.
    ar_ema_rate = math.pow(0.5,1/config.ar_ema_halflife) if config.ar_ema_halflife>0 else 0.
    enc_ema = copy.deepcopy(enc).requires_grad_(False).eval() if ae_ema_rate>0 else None
    dec_ema = copy.deepcopy(dec).requires_grad_(False).eval() if ae_ema_rate>0 else None
    freeze_ar_model = copy.deepcopy(ar_model).requires_grad_(False).eval() if config.train_ar else None
    ar_model_ema = copy.deepcopy(ar_model).requires_grad_(False).eval() if config.train_ar and ar_ema_rate>0 else None
    
    if config.ae_ckpt_path != "":
        fabric.print(f'resume ae ema from ckpt {config.ae_ckpt_path}')
        ckpt_states = fabric.load(config.ae_ckpt_path)
        if enc_ema is not None and ckpt_states.get('enc_ema', None) is not None:
            enc_ema.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ckpt_states['enc_ema'].items() })
        if dec_ema is not None and ckpt_states.get('dec_ema', None) is not None:
            dec_ema.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ckpt_states['dec_ema'].items() })
        print("loaded ae ema")

    if config.ar_ckpt_path!="" and config.load_ar_model_ema and ar_model_ema is not None and ar_ckpt_states.get('ar_model_ema', None) is not None:
        ar_model_ema.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ar_ckpt_states['ar_model_ema'].items() })
        print("loaded ar ema")
    if config.ar_ckpt_path!="" and config.load_ar_model_ema and freeze_ar_model is not None and ar_ckpt_states.get('freeze_ar_model', None) is not None:
        freeze_ar_model.load_state_dict({k.split('_orig_mod.')[-1]: v for k, v in ar_ckpt_states['freeze_ar_model'].items() })
        print("loaded frozen ar")

    # for inference
    enc_raw = enc
    dec_raw = dec
    ar_model_raw = ar_model
    enc_ema_raw = enc_ema
    dec_ema_raw = dec_ema
    ar_model_ema_raw = ar_model_ema

    if fabric.is_global_zero:
        fabric.print("Calculating model stats...")
        print_model_stats(fabric,config, enc_raw, dec_raw, ar_model_raw)
        root_dir = Path(save_dir)
        root_dir.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(config, root_dir / "config.yaml", resolve=True)
        img_dir  = Path(save_dir+ "/images")
        img_dir.mkdir(exist_ok=True, parents=True)
        ckpt_dir  = Path(save_dir+ "/ckpts")
        ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    fabric.barrier()

    if config.torch_compile:
        if enc is not None and config.compile_ae:
            enc = torch.compile(enc)
        if enc_ema is not None and config.compile_ae:
            enc_ema = torch.compile(enc_ema)
        if dec is not None and config.compile_ae:
            dec = torch.compile(dec)
        if dec_ema is not None and config.compile_ae:
            dec_ema = torch.compile(dec_ema)
        if ar_model is not None and config.compile_ar:
            ar_model = torch.compile(ar_model)
        if ar_model_ema is not None and config.compile_ar:
            ar_model_ema = torch.compile(ar_model_ema)
        if freeze_ar_model is not None and config.compile_ar:
            freeze_ar_model= torch.compile(freeze_ar_model)
        if gan_loss is not None and config.compile_gan:
            gan_loss = torch.compile(gan_loss)

    opt_enc = torch.optim.AdamW(enc.parameters() , lr=config.lr_enc, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay_enc) if config.train_ae else None
    opt_ar = torch.optim.AdamW(ar_model.parameters(), lr=config.lr_ar, betas=(config.ar_beta1, config.ar_beta2), weight_decay=config.weight_decay_ar) if config.train_ar else None
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=config.lr_dec, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay_dec) if config.train_ae else None
    opt_gan_loss = torch.optim.AdamW(gan_loss.parameters(), lr=config.lr_gan_loss, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay_gan) if config.use_gan_loss else None
    if  config.resume_optimizer:
        if config.train_ae:       
            fabric.print("resumed AE optimizer")
            opt_enc.load_state_dict(ckpt_states['opt_enc'])
            opt_dec.load_state_dict(ckpt_states['opt_dec'])
        if config.train_ar:
            fabric.print("resumed AR optimizer")
            ar_optimizer_states = ar_ckpt_states if config.ar_ckpt_path else ckpt_states
            opt_ar.load_state_dict(ar_optimizer_states['opt_ar'])
        if config.use_gan_loss:
            fabric.print("resumed GAN optimizer")
            opt_gan_loss.load_state_dict(ckpt_states['opt_gan_loss'])

    # Datasets
    dataset = ImageFolderDataset(path=config['data_dir'],
        resolution=config.image_size,
        use_label=config.use_label,
        max_size=None,
        xflip=config.xflip,
        crop_type=config.crop_type,
        deterministic_crop=bool(getattr(config, "deterministic_crop", False)),
        crop_seed=int(config.seed) if config.seed is not None else 0,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size//num_devices,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=worker_init,
        generator=dl_generator,
    )
    eval_loader=None
    if config.eval_data_dir is not None:
        eval_dataset = ImageFolderDataset(path=config['eval_data_dir'],
            resolution=config.image_size,
            use_label=config.use_label,
            max_size=None,
            xflip=False,
            crop_type=config.eval_crop_type,
            deterministic_crop=bool(getattr(config, "deterministic_crop", False)),
            crop_seed=int(config.seed) if config.seed is not None else 0,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size//num_devices,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init,
            generator=dl_generator,
        )
    
    # Setup Model, Optimizer, DataLoader with Fabric
    if config.train_ae:
        enc, opt_enc = fabric.setup(enc, opt_enc)
        dec, opt_dec = fabric.setup(dec, opt_dec)
        if enc_ema is not None:
            enc_ema = enc_ema.to(fabric.device)
        if dec_ema is not None:
            dec_ema = dec_ema.to(fabric.device)
    if config.train_ar:
        ar_model, opt_ar = fabric.setup(ar_model, opt_ar)
        if ar_model_ema is not None:
            ar_model_ema = ar_model_ema.to(fabric.device)
        if freeze_ar_model is not None:
            freeze_ar_model = freeze_ar_model.to(fabric.device)
    if config.use_gan_loss:
        gan_loss, opt_gan_loss = fabric.setup(gan_loss, opt_gan_loss)

    train_loader = fabric.setup_dataloaders(train_loader)
    train_loader = InfiniteIterator(train_loader)
    if eval_loader is not None:
        eval_loader = fabric.setup_dataloaders(eval_loader)


    if config.train_ar:
        ar_model.mark_forward_method('sampling')

    global_step = 0

    phases = parse_phases(config.phases)
    fabric.print("Training Phases: ", phases)
    total_phase_steps = sum(phase.num_steps for phase in phases)
    total_phase = len(phases)

    prev_phase_idx=-1
    phase_step_accum = list(itertools.accumulate([phase.num_steps for phase in phases]))
    pbar = tqdm(total=total_phase_steps, disable=not fabric.is_global_zero, dynamic_ncols=True, file=sys.stdout)
    pbar.set_description(f"Total Steps {total_phase_steps}")
    data_buffer = []
    while global_step < total_phase_steps:
        # Assign gradient and optimizer based on current phase and target
        phase_idx, inner_idx, target, internel_step = get_phase(global_step, phases, phase_step_accum, config)
        if not any(target):
            phase_idx, inner_idx, target, internel_step = get_phase(global_step+1, phases, phase_step_accum, config)
        if phase_idx != prev_phase_idx:
            if config.lr_scheduler != 'none':
                scheduler_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=phases[phase_idx].num_steps/len(phases[phase_idx].internal_steps),eta_min=config.lr_enc_min)  if config.train_ae else None
                scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=phases[phase_idx].num_steps/len(phases[phase_idx].internal_steps),eta_min=config.lr_dec_min)  if config.train_ae else None
                scheduler_ar = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ar, T_max=phases[phase_idx].num_steps/len(phases[phase_idx].internal_steps),eta_min=config.lr_ar_min)  if config.train_ar else None
                scheduler_gan_loss = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gan_loss, T_max=phases[phase_idx].num_steps/len(phases[phase_idx].internal_steps),eta_min=config.lr_gan_loss_min)  if config.use_gan_loss else None
            else:
                scheduler_enc = None
                scheduler_dec = None
                scheduler_ar = None
                scheduler_gan_loss = None


        enc_grad = target.DO_AE or target.DO_PRIOR_ENC
        dec_grad = target.DO_AE
        prior_grad = target.DO_PRIOR_AR
        disc_grad = target.DO_GAN_D 
        toogle_require_grad(enc, enc_grad)
        toogle_require_grad(dec, dec_grad)
        toogle_require_grad(ar_model, prior_grad)
        toogle_require_grad(gan_loss, disc_grad)
        opts = []
        if enc_grad:
            opts.append((enc, opt_enc, scheduler_enc))
        if dec_grad:
            opts.append((dec, opt_dec, scheduler_dec))
        if prior_grad:
            opts.append((ar_model, opt_ar, scheduler_ar))
        if disc_grad:
            opts.append((gan_loss, opt_gan_loss, scheduler_gan_loss))

        # input
        # Clear buffer at start of cycle
        if internel_step == 0:
            data_buffer = []

        if inner_idx == 0:
            batch = next(train_loader)
            data_buffer.append(batch)
        else:
            if len(data_buffer) > 0:
                batch = data_buffer.pop(0)
            else:
                batch = next(train_loader)

        imgs, labels = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
        imgs = img_unint8_to_norm(imgs)

        # no labels or drop labels
        uncond_labels = F.one_hot(torch.full((imgs.shape[0],), config.num_classes - 1, device=fabric.device, dtype=torch.long), num_classes=config.num_classes).float()

        if labels is None or len(labels)==0 or not config.use_label:
            labels = uncond_labels
        elif config.label_drop_prob > 0:
            labels = torch.where(torch.rand(imgs.shape[0], device=fabric.device) < config.label_drop_prob, uncond_labels, labels)

        x = patchify(imgs, config.patch_size)

        
        # forward, build need computation graph for gradient calculation
        with torch.set_grad_enabled(enc_grad):
            quant, idx, one_hot, vqloss = enc(x, labels, training=enc_grad)
        
        if target.DO_AE:
            
            # For Diffusion/FM decoder
            t_fm,xt,noise,v = None,None,None,None
            if config.use_noise_query:
                t_fm = torch.rand((x.shape[0],), device=fabric.device) 
                xt, noise, v = make_noise_image_patch_query(x, t_fm, config.z_len) 

            # For Tail Dropout
            attn_bias = None
            if config.use_tail_dropout and global_step>config.tail_dropout_start:
                if config.tied_timestep:
                    t_tail_drop = t_fm
                else:
                    t_tail_drop = torch.rand((x.shape[0],), device=fabric.device)
                    no_drop_t = torch.ones_like(t_tail_drop)
                    t_tail_drop = torch.where(torch.rand_like(t_tail_drop)<=config.tail_dropout_p,t_tail_drop,no_drop_t)
                attn_bias = build_taildrop_attn_mask(t=t_tail_drop,context_len=int(quant.shape[1]),query_len=config.x_len,
                                    context_type=config.context_type,t2k=config.tail_dropout_t2k,
                                    device=fabric.device,
                                    diti_scheduler=diti_scheduler,
                                                    ) if config.use_tail_dropout else None

            decoded = dec(quant, labels, attn_mask=attn_bias, query_tokens=xt, t=t_fm)

            v_hat = decoded if config.l2_v_target else None
            x_hat = xt - decoded * t_fm[:,None,None] if config.l2_v_target else decoded

            imgs_hat = unpatchify(x_hat, config.image_size, config.patch_size)
            
            l2_target = v if config.l2_v_target else x
            l2_input = v_hat if config.l2_v_target else x_hat

        if target.DO_PRIOR_ENC or target.DO_PRIOR_AR:
            with torch.set_grad_enabled(prior_grad):
                ar_logits = ar_model(idx,labels)

        if target.DO_PRIOR_ENC:
            with torch.no_grad():
                ema_ar_logits = freeze_ar_model(idx,labels)
                with torch.amp.autocast('cuda',enabled=False):
                    ar_logits_detach = ar_logits.detach().float()
                    temp_low = config.prior_ar_temp_low
                    temp_high = config.prior_ar_temp_high
                    steps = config.z_len
                    if config.prior_ar_temp_schedule == 'cosine':
                        t = torch.linspace(0, 1, steps, device=fabric.device)
                        temp = temp_low + 0.5 * (temp_high - temp_low) * (1 - torch.cos(math.pi * t))
                    elif config.prior_ar_temp_schedule == 'linear':
                        temp = torch.linspace(temp_low, temp_high, steps, device=fabric.device)
                    elif config.prior_ar_temp_schedule == 'constant':
                        temp = torch.full((steps,), temp_low, device=fabric.device)
                    else:
                        raise NotImplementedError(f"Unknown schedule: {config.prior_ar_temp_schedule}")

                    b, n = ema_ar_logits.shape[0], ema_ar_logits.shape[1]
                    temp = temp.view(1, n, 1).expand(b, n, 1)

                    ema_ar_logits = ema_ar_logits.detach().float() / temp
                    prior_ema_ar_loss = F.cross_entropy(ema_ar_logits.reshape(-1, ema_ar_logits.shape[-1]).float(), idx.reshape(-1)) 

                    ema_ar_log_probs= torch.log_softmax(ema_ar_logits, dim=-1)
                    ar_log_probs = torch.log_softmax(ar_logits_detach, dim=-1)
                    entropy_ar_log_probs = -(ar_log_probs.exp() * ar_log_probs).sum(dim=-1)
                    entropy_ema_ar_log_probs = -(ema_ar_log_probs.exp() * ema_ar_log_probs).sum(dim=-1)
                    log_diff = ar_log_probs - ema_ar_log_probs


        # caculate the loss
        l2_loss = config.l2_weight * F.mse_loss(l2_input, l2_target) if target.DO_L2 else 0.
        l1_loss = config.l1_weight * F.l1_loss(x_hat, x) if target.DO_L1 else 0.
        lpips_loss = config.lpips_weight * perceptual_loss(imgs_hat, imgs).mean() if target.DO_LPIPS else 0.
        ae_loss = l2_loss + l1_loss + lpips_loss

        gan_G_loss, gan_G_loss_dict = gan_loss(imgs_hat, imgs, global_step=global_step, loss='G') if target.DO_GAN_G else (0., {})
        gan_D_loss, gan_D_loss_dict = gan_loss(imgs_hat, imgs, global_step=global_step, loss='D') if target.DO_GAN_D else (0., {})
        gan_G_loss_weight = (config.gan_G_weight * min(1., gan_loss.calculate_adaptive_weight(ae_loss,gan_G_loss,dec.dec.out.weight))) if target.DO_GAN_G and config.disc_adaptive_weight else config.gan_G_weight
        gan_G_loss = gan_G_loss * gan_G_loss_weight
        gan_D_loss = gan_D_loss * config.gan_G_weight
        correct_tokens = (ar_logits.argmax(dim=-1) == idx).detach().float().sum().item() if target.DO_PRIOR_AR else 0.
        total_tokens = torch.tensor(idx.numel()).item() if target.DO_PRIOR_AR else 0.
        correct_token_rate = correct_tokens / total_tokens if target.DO_PRIOR_AR else 0.
        prior_enc_loss = config.prior_enc_weight * (one_hot * log_diff.detach()).mean() if target.DO_PRIOR_ENC else 0.
        prior_ar_loss = F.cross_entropy(ar_logits.reshape(-1, ar_logits.shape[-1]).float(), idx.reshape(-1)) if target.DO_PRIOR_AR else 0.

        loss = ae_loss + prior_enc_loss + prior_ar_loss + gan_G_loss + gan_D_loss + vqloss 

        # backward and optimization
        fabric.backward(loss)
        
        for model, opt, scheduler in opts:
            if model is not None and opt is not None:
                zero_nan_gradients(model)
                if config.grad_clip>0:
                    fabric.clip_gradients(model, opt, max_norm=config.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step() 
        if config.train_ae:
            if enc_grad:
                ema_update(enc, enc_ema, ae_ema_rate)
            if dec_grad:
                ema_update(dec, dec_ema, ae_ema_rate)
            if config.use_gan_loss and disc_grad:
                ema_update(gan_loss, gan_loss_ema, ae_ema_rate)
        if config.train_ar and prior_grad:
            ema_update(ar_model, ar_model_ema, ar_ema_rate)

        #visualization
        if (global_step + 1) % config.visualize_freq == 0:
            grid = x[:config.visualize_img_num]
            if not config.ema_reconstruction:
                if config.train_ae:
                    toogle_train_eval(enc_raw,train=False) 
                    toogle_train_eval(dec_raw,train=False) 
                eval_enc = enc_raw
                eval_dec = dec_raw
            else:
                eval_enc = enc_ema_raw
                eval_dec = dec_ema_raw
            if not config.ema_sampling:
                if config.train_ar:
                    toogle_train_eval(ar_model_raw,train=False) 
                eval_ar = ar_model_raw
            else:
                eval_ar = ar_model_ema_raw
            with torch.no_grad():
                noise_t = torch.ones((x.shape[0],), device=fabric.device) if config.use_noise_query else None
                if config.train_ae:
                    noise_recon = torch.randn_like(x) if config.use_noise_query else None
                    if config.use_noise_query and config.diffusion_decoder:
                        all_recon_images,_ = euler_recon_loop(eval_enc, eval_dec, x, labels, config=config, diti_scheduler=diti_scheduler)
                    else:
                        decoded_recon,_ = reconstruction(eval_enc, eval_dec, x, labels, query_tokens=noise_recon, t=noise_t)
                        all_recon_images = decoded_recon if not config.l2_v_target else noise_recon-decoded_recon
                    grid = torch.cat([grid, all_recon_images[:config.visualize_img_num]], dim=0)

                if config.train_ar:
                    noise_sample = torch.randn_like(x) if config.use_noise_query else None
                    if config.use_noise_query and config.diffusion_decoder:
                        all_sample_images = euler_sample_loop(eval_enc, eval_dec, eval_ar, bz=x.shape[0], 
                                            class_label=labels,temperature=config.temperature, 
                                            topK=config.topK, topP=config.topP, 
                                            cfg=config.cfg, cfg_schedule=config.cfg_schedule, cfg_power=config.cfg_power, 
                                            cache_kv=config.cache_kv, config=config, diti_scheduler=diti_scheduler)
                    else:
                        decoded_sample = sampling(eval_enc, eval_dec, eval_ar, bz=x.shape[0], 
                                                class_label=labels,temperature=config.temperature, 
                                                topK=config.topK, topP=config.topP, 
                                                cfg=config.cfg, cfg_schedule=config.cfg_schedule, cfg_power=config.cfg_power, 
                                                cache_kv=config.cache_kv, config=config, t=noise_t ,query_tokens=noise_sample)
                        all_sample_images = decoded_sample if not config.l2_v_target else noise_sample-decoded_sample
                    grid = torch.cat([grid, all_sample_images[:config.visualize_img_num]], dim=0)
                grid = img_denormalize(unpatchify(grid, config.image_size, config.patch_size))
                grid = torchvision.utils.make_grid(grid, nrow=config.visualize_img_num, normalize=False)
                fabric.log_dict({"visualization":wandb.Image(grid),'global_step':global_step+1})
            if not config.ema_reconstruction and config.train_ae:
                toogle_train_eval(enc_raw,train=True) 
                toogle_train_eval(dec_raw,train=True) 
            if not config.ema_sampling and config.train_ar:
                toogle_train_eval(ar_model_raw,train=True) 
           
        # eval and save ckpt
        rFID = 0.0
        gFID = 0.0
        if eval_loader is not None and (global_step + 1) % config.eval_freq == 0:
             
            sample_cached_path = os.path.join(save_dir, "sample_images.npz")
            recon_cached_path = os.path.join(save_dir, "recon_images.npz")
            gt_cache_path = config.eval_fid_ref_path

            samples_buf = []
            recons_buf = []
            if not config.ema_reconstruction:
                if config.train_ae:
                    toogle_train_eval(enc_raw,train=False) 
                    toogle_train_eval(dec_raw,train=False) 
                eval_enc = enc_raw
                eval_dec = dec_raw
            else:
                eval_enc = enc_ema_raw
                eval_dec = dec_ema_raw
            
            if not config.ema_sampling:
                if config.train_ar:
                    toogle_train_eval(ar_model_raw, train=False)
                eval_ar = ar_model_raw
            else:
                eval_ar = ar_model_ema_raw


            with torch.no_grad():

                fabric.barrier()
                for i, batch in enumerate(tqdm(eval_loader,disable=not fabric.is_global_zero,dynamic_ncols=True,file=sys.stdout,desc="Evaluating")):
                    imgs, labels = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
                    x = img_unint8_to_norm(imgs)
                    x = patchify(x, config.patch_size)
                    uncond_labels = F.one_hot(torch.full((imgs.shape[0],), config.num_classes - 1, device=fabric.device, dtype=torch.long), num_classes=config.num_classes).float()
                    noise_t = torch.ones((labels.shape[0],), device=fabric.device) if config.use_noise_query else None

                    if labels is None or len(labels)==0 or not config.use_label:
                        labels = uncond_labels
                    idx = None
                    noise_recon = torch.randn_like(x) if config.use_noise_query else None

                    if config.train_ae:
                        if config.use_noise_query and config.diffusion_decoder:
                            recon_images, idx = euler_recon_loop(eval_enc, eval_dec, x, labels, config=config, diti_scheduler=diti_scheduler)
                        else:
                            decoded_recon, idx = reconstruction(eval_enc, eval_dec, x, labels, query_tokens=noise_recon,t = noise_t)
                            recon_images = decoded_recon if not config.l2_v_target else noise_recon-decoded_recon

                        recon_images = img_norm_to_uint8(unpatchify(recon_images, config.image_size, config.patch_size))
                        if fabric.world_size>1:
                            recon_images = fabric.all_gather(recon_images) 
                            recon_images = recon_images.flatten(0,1)
                        recons_buf.append(recon_images.permute(0,2,3,1).cpu().numpy())
                    if config.train_ar:
                        if config.use_noise_query and config.diffusion_decoder:
                            sample_images = euler_sample_loop(eval_enc, eval_dec, eval_ar, bz=x.shape[0], 
                                                class_label=labels,temperature=config.temperature, 
                                                topK=config.topK, topP=config.topP, 
                                                cfg=config.cfg, cfg_schedule=config.cfg_schedule, cfg_power=config.cfg_power, 
                                                cache_kv=config.cache_kv, config=config, diti_scheduler=diti_scheduler)
                        else:
                            noise_sample = torch.randn_like(x) if config.use_noise_query else None
                            decoded_sample = sampling(eval_enc, eval_dec, eval_ar, 
                                            bz=x.shape[0], class_label=labels,temperature=config.temperature, 
                                            topK=config.topK, topP=config.topP, cfg=config.cfg, cfg_schedule=config.cfg_schedule, 
                                            cfg_power=config.cfg_power, cache_kv=config.cache_kv, config=config,t=noise_t,query_tokens=noise_sample)
                            sample_images = decoded_sample if not config.l2_v_target else noise_sample-decoded_sample

                        sample_images = img_norm_to_uint8(unpatchify(sample_images, config.image_size, config.patch_size))
                        if fabric.world_size>1:
                            sample_images = fabric.all_gather(sample_images) 
                            sample_images = sample_images.flatten(0,1)
                        samples_buf.append(sample_images.permute(0,2,3,1).cpu().numpy())

        
                fabric.barrier()
                if fabric.is_global_zero:
                    if config.train_ar:
                        np.savez_compressed(sample_cached_path, np.concatenate(samples_buf,axis=0))
                    if config.train_ae:
                        np.savez_compressed(recon_cached_path, np.concatenate(recons_buf,axis=0))
                    
                    if config.train_ar:
                        gFID = adm_fid_evaluator(sample_cached_path, gt_cache_path, config, fabric)
                    if config.train_ae:
                        rFID = adm_fid_evaluator(recon_cached_path, gt_cache_path, config, fabric)
                fabric.barrier()


            if not config.ema_reconstruction and config.train_ae:
                toogle_train_eval(enc_raw,train=True) 
                toogle_train_eval(dec_raw,train=True) 
            if not config.ema_sampling and config.train_ar:
                toogle_train_eval(ar_model_raw,train=True) 

        # metrics
        if (global_step +1) % config.log_freq == 0:
            # current learning rates
            lr_enc = opt_enc.param_groups[0]['lr'] if opt_enc is not None else 0.
            lr_dec = opt_dec.param_groups[0]['lr'] if opt_dec is not None else 0.
            lr_ar = opt_ar.param_groups[0]['lr'] if opt_ar is not None else 0.
            lr_gan_loss = opt_gan_loss.param_groups[0]['lr'] if opt_gan_loss is not None else 0.
            metrics = {
                        'Phase':phase_idx,
                        'ae_loss/l1_loss':l1_loss,
                        'ae_loss/l2_loss':l2_loss,
                        'ae_loss/lpips_loss':lpips_loss,
                        'ae_loss/loss':ae_loss, 
                        'ae_loss/vqloss': vqloss,
                        'ae_loss/gan_G_loss':gan_G_loss,
                        'ae_loss/gan_D_loss':gan_D_loss,
                        'prior_loss/logq-logp':log_diff.mean() if target.DO_PRIOR_ENC else 0., 
                        'prior_loss/logq':ar_log_probs.mean() if target.DO_PRIOR_ENC else 0., 
                        'prior_loss/logp':ema_ar_log_probs.mean() if target.DO_PRIOR_ENC else 0., 
                        'prior_loss/entropy_q':entropy_ar_log_probs.mean() if target.DO_PRIOR_ENC else 0.,
                        'prior_loss/entropy_p':entropy_ema_ar_log_probs.mean() if target.DO_PRIOR_ENC else 0.,
                        'prior_loss/ar_loss':prior_ar_loss, 
                        'prior_loss/ema_ar_loss':prior_ema_ar_loss  if target.DO_PRIOR_ENC else 0.,
                        'prior_loss/correct_token_rate': correct_token_rate,
                        'FID/rFID':rFID,
                        'FID/gFID':gFID,
                        'lr/lr_enc': lr_enc,
                        'lr/lr_dec': lr_dec,
                        'lr/lr_ar': lr_ar,
                        'lr/lr_gan_loss': lr_gan_loss,
                      }
            gan_loss_dict = {**gan_G_loss_dict, **gan_D_loss_dict, 'gan_G_loss_weight':gan_G_loss_weight}
            metrics.update(gan_loss_dict)
            metrics = {k: fabric.all_reduce(v).item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items() }
            metrics_logger = {k: v for k, v in metrics.items() if v!=0. }
            metrics_4f = {k: f"{v:.4f}" if k!='Phase' else int(v) for k, v in metrics.items() }
            fabric.log_dict(metrics_logger,step=global_step+1)
            pbar.set_postfix(metrics_4f)
            pbar.update(config.log_freq)

        if config.save_ckpt and (global_step==0 or (global_step+1) % config.ckpt_freq == 0):
            ckpt_name = f'Phase={phase_idx}-Step={global_step+1}-rFID={rFID:.4f}-gFID={gFID:.4f}' if rFID is not None and gFID is not None else "_".join([f'{k}={v:.4f}' for k, v in metrics.items()]) 
            ckpt_path = f'{save_dir}/ckpts/{ckpt_name}.ckpt'
            fabric.save(ckpt_path, {'enc':enc,
                        'ar_model':ar_model,
                        'freeze_ar_model': freeze_ar_model,
                        'dec':dec,
                        'gan_loss':gan_loss,
                        'enc_ema':enc_ema,
                        'dec_ema':dec_ema,
                        'ar_model_ema':ar_model_ema,
                        'opt_enc':opt_enc,
                        'opt_ar':opt_ar,
                        'opt_dec':opt_dec,
                        'opt_gan_loss':opt_gan_loss})
        global_step += 1
        prev_phase_idx = phase_idx
    if config.save_ckpt:
        ckpt_path = f'{save_dir}/ckpts/last.ckpt'
        fabric.save(ckpt_path, {'enc':enc,
                    'ar_model':ar_model,
                    'dec':dec,
                    'gan_loss':gan_loss,
                    'freeze_ar_model':freeze_ar_model,
                    'enc_ema':enc_ema,
                    'dec_ema':dec_ema,
                    'ar_model_ema':ar_model_ema,
                    'opt_enc':opt_enc,
                    'opt_ar':opt_ar,
                    'opt_dec':opt_dec,
                    'opt_gan_loss':opt_gan_loss})

    logger.finalize("Training completed")

if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    cli_conf = OmegaConf.from_cli()

    config_path = cli_conf.get("--config")
    conf = OmegaConf.load(config_path)
    if cli_conf is not None:
        for k, v in cli_conf.items():
            OmegaConf.update(conf, k, v)

    train(conf)

