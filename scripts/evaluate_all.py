"""Evaluate CIFAR-10 checkpoints with independent per-GPU seeds."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

# Method names match configuration prefixes and checkpoint filenames.
CONFIGS = {
    "baseline": "baseline_ar.yaml",
    "tail_dropout": "tail_dropout_ar.yaml",
    "flow_matching": "flow_matching_ar.yaml",
    "flow_matching_tail_dropout": "flow_matching_tail_dropout_ar.yaml",
    "flow_matching_tied_tail_dropout": "flow_matching_tied_tail_dropout_ar.yaml",
    "war": "war_ar.yaml"
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_free(gpus):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    devices = {int(row[0]): (int(row[1]), int(row[2]))
               for row in csv.reader(io.StringIO(output))}
    busy = {gpu: devices.get(gpu) for gpu in gpus
            if gpu not in devices or devices[gpu][0] > 100 or devices[gpu][1] > 0}
    if busy:
        raise RuntimeError(f'Requested GPUs are not free: {busy}')


def save_summary(output, rows):
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if rows:
        with (output / 'results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--output', default='outputs/cifar10-nonlegacy')
    parser.add_argument('--methods', nargs='+', choices=CONFIGS,
                        help='Optional subset, e.g. baseline war')
    parser.add_argument('--resume', action='store_true', help='Reuse matching completed evaluations')
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error('GPU IDs must be nonempty and distinct')
    configs = {key: ROOT / 'configs/cifar10' / name for key, name in CONFIGS.items()}
    for config in configs.values():
        if not config.is_file():
            raise FileNotFoundError(config)
    if args.methods:
        if set(args.methods) - configs.keys():
            parser.error('Unknown method')
        configs = {key: path for key, path in configs.items() if key in args.methods}
    checkpoints = {key: ROOT / 'checkpoints' / f'{key}.pt' for key in configs}
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Place the downloaded checkpoint at {checkpoint}')
    reference = ROOT / 'refs/cifar10-32x32_local.npz'
    reference_hash = sha256(reference)
    if reference_hash != '5493968c4f29b2878f64d6a640351f377e32e86553db56eca8147a0ae50d93f2':
        raise ValueError('The CIFAR-10 FID reference checksum does not match')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), PYTHONDONTWRITEBYTECODE='1',
               WANDB_MODE='disabled', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TF_CPP_MIN_LOG_LEVEL='2')
    rows = []
    for key, config in configs.items():
        checkpoint = checkpoints[key]
        checkpoint_hash, config_hash = sha256(checkpoint), sha256(config)
        case = output / key
        result_path = case / 'metrics.json'
        if result_path.exists() and not args.resume:
            raise FileExistsError(result_path)
        if not result_path.exists():
            check_free(gpus)
            case.mkdir(exist_ok=True)
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={len(gpus)}', 'evaluate.py', '--config', str(config),
                   '--checkpoint', str(checkpoint), '--num-samples', '50000', '--batch-size', '64',
                   '--seed', '42', '--no-legacy', '--output', str(case), '--fid-reference', str(reference)]
            request = dict(command=cmd, checkpoint_sha256=checkpoint_hash,
                           config_sha256=config_hash, legacy=False)
            (case / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
            print(f'START {key}', flush=True)
            started = time.monotonic()
            with (case / 'run.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'DONE {key} in {time.monotonic() - started:.1f}s', flush=True)
        result = json.loads(result_path.read_text())
        request = json.loads((case / 'request.json').read_text())
        if request['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError(f'{key}: checkpoint changed since the recorded evaluation')
        expected = dict(legacy=False, num_samples=50000, seed=42, world_size=len(gpus),
                        batch_size=64, ar_ema=True, precision='32-true', decoder='single_step',
                        config_sha256=config_hash, reference_sha256=reference_hash)
        if any(result.get(name) != value for name, value in expected.items()):
            raise ValueError(f'{key}: recorded evaluation settings do not match')
        row = dict(method=key, nonlegacy_gfid=result['fid'], unique_images=result['unique_images'],
                   num_samples=result['num_samples'], legacy=False,
                   checkpoint_sha256=checkpoint_hash, config_sha256=config_hash,
                   reference_sha256=reference_hash)
        rows.append(row)
        save_summary(output, rows)
        print(f'RESULT {key}: {result["fid"]:.8f}', flush=True)
    print(f'COMPLETE: {len(rows)} non-legacy evaluations', flush=True)


if __name__ == '__main__':
    main()
