# wAR-Tok: Learning Discrete Autoregressive Priors with Wasserstein Gradient Flow

CIFAR-10 implementation of [our paper](https://arxiv.org/abs/2605.06148).

## Features

- Discrete image tokenization with Wasserstein gradient flow regularization.
- Transformer encoder and decoder with cross-attention, RMSNorm and GEGLU.
- Six variants: baseline, tail dropout, flow matching, FM + tail dropout, FM + tied tail dropout, and wAR-Tok.
- Two-stage training: tokenizer training followed by an autoregressive prior.
- Mixed precision, multi-GPU training and optional W&B logging.

## Quick Start

### Installation

Use Python 3.12 and CUDA. The training environment uses PyTorch 2.8.0 / CUDA 12.8.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

```bash
python scripts/prepare_cifar10.py --download
```

This prepares the CIFAR-10 training split under `data/cifar10`. Download the [FID reference](https://huggingface.co/Zyriix/wARTok/resolve/main/refs/cifar10-32x32_local.npz?download=true) to `refs/cifar10-32x32_local.npz`.

### Training

The default configuration uses 8 GPUs, global batch size 512 and BF16. Each method has a `*_tokenizer.yaml` and a `*_ar.yaml` in [configs/cifar10](configs/cifar10): train the tokenizer for 400K steps, then the AR prior for 200K steps.

**Stage 1: Train the tokenizer**

```bash
python train.py --config=configs/cifar10/war_tokenizer.yaml
```

**Stage 2: Train the AR prior**

```bash
python train.py --config=configs/cifar10/war_ar.yaml \
  ae_ckpt_path=./outputs/war_tokenizer/000/ckpts/last.ckpt
```

The AR configuration defaults to the matching tokenizer's `000/ckpts/last.ckpt`. Set `WANDB_MODE=online` to enable experiment logging.

### Evaluation

Use the matching tokenizer configuration for reconstruction and the AR configuration for generation. FM reconstruction uses 25 Euler steps; generation uses a single decoder step. Evaluation uses FP32; generation uses AR EMA, temperature 1, CFG 0 and no top-k/top-p filtering.

**Reconstruction**

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config configs/cifar10/war_tokenizer.yaml --checkpoint checkpoints/war.pt \
  --mode reconstruct --data data/cifar10 --num-samples 50000 \
  --output outputs/war-reconstruction --fid-reference refs/cifar10-32x32_local.npz
```

**Generation**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 evaluate.py \
  --config configs/cifar10/war_ar.yaml --checkpoint checkpoints/war.pt \
  --num-samples 50000 --batch-size 64 --seed 42 --no-legacy \
  --output outputs/war-generation --fid-reference refs/cifar10-32x32_local.npz
```

Non-legacy mode (`--no-legacy`) uses a different seed on each GPU and generally yields [lower gFID scores](#checkpoints), without changing the relative ranking of the methods or the conclusions supported by the results.

To evaluate all six checkpoints with `--no-legacy`:

```bash
python scripts/evaluate_all.py --gpus 0,1,2,3,4,5,6,7 \
  --output outputs/cifar10-nonlegacy
```

Use `--methods baseline war` for a subset or `--resume` to reuse completed evaluations.

### Exporting Checkpoints

```bash
python scripts/export_checkpoint.py --source path/to/raw-ar.ckpt \
  --config configs/cifar10/war_ar.yaml --output checkpoints/war.pt
```

The export includes tokenizer and raw/EMA AR weights without optimizer states.

## Checkpoints

Name checkpoints after their configuration prefix (e.g. `baseline_ar.yaml` → `checkpoints/baseline.pt`, `war_ar.yaml` → `checkpoints/war.pt`); each includes the tokenizer and raw/EMA AR weights. Hugging Face repository: [wARTok](https://huggingface.co/Zyriix/wARTok).

| Method | rFID | Historical gFID | Non-legacy gFID | Download |
|---|---:|---:|---:|---|
| [Baseline](configs/cifar10/baseline_ar.yaml) | 2.3015 | 52.5343 | 49.4191 | [Download](https://huggingface.co/Zyriix/wARTok/resolve/main/checkpoints/baseline.pt?download=true) |
| [Tail Dropout](configs/cifar10/tail_dropout_ar.yaml) | 3.3567 | 23.3713 | 19.8997 | [Download](https://huggingface.co/Zyriix/wARTok/resolve/main/checkpoints/tail_dropout.pt?download=true) |
| [FM](configs/cifar10/flow_matching_ar.yaml) | 3.4194 | 38.6309 | 35.8593 | [Download](https://huggingface.co/Zyriix/wARTok/resolve/main/checkpoints/flow_matching.pt?download=true) |
| [FM + Tail](configs/cifar10/flow_matching_tail_dropout_ar.yaml) | 4.4247 | 22.7613 | 19.0569 | [Download](https://huggingface.co/Zyriix/wARTok/resolve/main/checkpoints/flow_matching_tail_dropout.pt?download=true) |
| [FM + Tail + Tied](configs/cifar10/flow_matching_tied_tail_dropout_ar.yaml) | 6.4911 | 44.7971 | 41.1334 | [Download](https://huggingface.co/Zyriix/wARTok/resolve/main/checkpoints/flow_matching_tied_tail_dropout.pt?download=true) |
| [wAR-Tok](configs/cifar10/war_ar.yaml) | 3.0648 | 11.3059 | 7.3749 | [Download](https://huggingface.co/Zyriix/wARTok/resolve/main/checkpoints/war.pt?download=true) |

Download the [FID reference](https://huggingface.co/Zyriix/wARTok/resolve/main/refs/cifar10-32x32_local.npz?download=true) to `refs/cifar10-32x32_local.npz`.

## Project Structure

```text
wAR/
├── configs/cifar10/   # Training configurations
├── checkpoints/      # Downloaded weights
├── scripts/          # Data preparation and evaluation tools
├── models.py         # Model architectures
├── train.py          # Tokenizer and AR training
├── evaluate.py       # Reconstruction and generation
├── dataset.py        # Data loading
├── lpips.py          # Perceptual loss
└── calc_fid.py        # ADM FID evaluation
```

## References

This project builds on [EDM](https://github.com/NVlabs/edm), [ADM](https://github.com/openai/guided-diffusion), and [LPIPS](https://github.com/richzhang/PerceptualSimilarity).

## License

Project-authored code is covered by [MIT](LICENSE). Third-party components retain their original licenses.
