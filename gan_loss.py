"""
Single-file GAN loss module with discriminators and perceptual loss.
"""

import functools
import hashlib
import math
import os
import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision import models
from torchvision.transforms import RandomCrop
from tqdm import tqdm

###############################################################################
# PatchGAN discriminator and ActNorm
###############################################################################


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix.

    Reference:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator.

        Args:
            input_nc (int): number of channels in input images.
            ndf (int): number of filters in the last conv layer.
            n_layers (int): number of conv layers in the discriminator.
            use_actnorm (bool): whether to use ActNorm instead of BatchNorm.
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm

        if isinstance(norm_layer, functools.partial):
            # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # output 1 channel prediction map
        sequence += [
            nn.Conv2d(
                ndf * nf_mult,
                1,
                kernel_size=kw,
                stride=1,
                padding=padw,
            )
        ]
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward."""
        return self.main(x)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x: torch.Tensor):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse:
            return self.reverse(x)
        if len(x.shape) == 2:
            x = x[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = x.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        h = self.scale * (x + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(x.shape[0]).to(x)
            return h, logdet

        return h

    def reverse(self, y: torch.Tensor):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is disabled by default. "
                    "Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(y)
                self.initialized.fill_(1)

        if len(y.shape) == 2:
            y = y[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = y / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


###############################################################################
# StyleGAN-like image discriminator
###############################################################################

try:
    from kornia.filters import filter2d
except Exception:
    filter2d = None


def leaky_relu(p: float = 0.2):
    return nn.LeakyReLU(p, inplace=True)


def exists(val):
    return val is not None


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if filter2d is None:
            return x
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.downsample = (
            nn.Sequential(
                Blur(),
                nn.Conv2d(filters, filters, 3, padding=1, stride=2),
            )
            if downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, channel_multiplier=1, image_size=256):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(math.log(image_size, 2))
        in_channel = channels[image_size]

        blocks = [nn.Conv2d(input_nc, in_channel, 3, padding=1), leaky_relu()]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(DiscriminatorBlock(in_channel, out_channel))
            in_channel = out_channel
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channel, channels[4], 3, padding=1),
            leaky_relu(),
        )
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            leaky_relu(),
            nn.Linear(channels[4], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


###############################################################################
# DiffAug (data augmentation for discriminator)
###############################################################################


class DiffAug(object):
    def __init__(self, prob: float = 1.0, cutout: float = 0.2):
        self.grids = {}
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout
        self.img_channels = -1
        self.last_blur_radius = -1
        self.last_blur_kernel_h = None
        self.last_blur_kernel_w = None

    def get_grids(self, B, x, y, dev):
        if (B, x, y) in self.grids:
            return self.grids[(B, x, y)]

        self.grids[(B, x, y)] = ret = torch.meshgrid(
            torch.arange(B, dtype=torch.long, device=dev),
            torch.arange(x, dtype=torch.long, device=dev),
            torch.arange(y, dtype=torch.long, device=dev),
            indexing="ij",
        )
        return ret

    def aug(self, BCHW: torch.Tensor, warmup_blur_schedule: float = 0) -> torch.Tensor:
        # warmup blurring
        if BCHW.dtype != torch.float32:
            BCHW = BCHW.float()
        if warmup_blur_schedule > 0:
            self.img_channels = BCHW.shape[1]
            sigma0 = (BCHW.shape[-2] * 0.5) ** 0.5
            sigma = sigma0 * warmup_blur_schedule
            blur_radius = math.floor(sigma * 3)  # 3-sigma is enough for Gaussian
            if blur_radius >= 1:
                if self.last_blur_radius != blur_radius:
                    self.last_blur_radius = blur_radius
                    gaussian = torch.arange(
                        -blur_radius, blur_radius + 1, dtype=torch.float32, device=BCHW.device
                    )
                    gaussian = gaussian.mul_(1 / sigma).square_().neg_().exp2_()
                    gaussian.div_(gaussian.sum())  # normalize
                    self.last_blur_kernel_h = (
                        gaussian.view(1, 1, 2 * blur_radius + 1, 1)
                        .repeat(self.img_channels, 1, 1, 1)
                        .contiguous()
                    )
                    self.last_blur_kernel_w = (
                        gaussian.view(1, 1, 1, 2 * blur_radius + 1)
                        .repeat(self.img_channels, 1, 1, 1)
                        .contiguous()
                    )

                BCHW = F.pad(BCHW, [blur_radius, blur_radius, blur_radius, blur_radius], mode="reflect")
                BCHW = F.conv2d(
                    input=BCHW, weight=self.last_blur_kernel_h, bias=None, groups=self.img_channels
                )
                BCHW = F.conv2d(
                    input=BCHW, weight=self.last_blur_kernel_w, bias=None, groups=self.img_channels
                )

        if self.prob < 1e-6:
            return BCHW
        trans, color, cut = torch.rand(3) <= self.prob
        trans, color, cut = trans.item(), color.item(), cut.item()
        B, dev = BCHW.shape[0], BCHW.device
        rand01 = torch.rand(7, B, 1, 1, device=dev) if (trans or color or cut) else None

        raw_h, raw_w = BCHW.shape[-2:]
        if trans:
            ratio = 0.125
            delta_h = round(raw_h * ratio)
            delta_w = round(raw_w * ratio)
            translation_h = rand01[0].mul(delta_h + delta_h + 1).floor().long() - delta_h
            translation_w = rand01[1].mul(delta_w + delta_w + 1).floor().long() - delta_w

            grid_B, grid_h, grid_w = self.get_grids(B, raw_h, raw_w, dev)
            grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h + 1)
            grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w + 1)
            bchw_pad = F.pad(BCHW, [1, 1, 1, 1, 0, 0, 0, 0])
            BCHW = (
                bchw_pad.permute(0, 2, 3, 1)
                .contiguous()[grid_B, grid_h, grid_w]
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        if color:
            BCHW = BCHW.add(rand01[2].unsqueeze(-1).sub(0.5))
            bchw_mean = BCHW.mean(dim=1, keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(bchw_mean)
            bchw_mean = BCHW.mean(dim=(1, 2, 3), keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(bchw_mean)

        if self.using_cutout and cut:
            ratio = self.cutout
            cutout_h = round(raw_h * ratio)
            cutout_w = round(raw_w * ratio)
            offset_h = rand01[5].mul(raw_h + (1 - cutout_h % 2)).floor().long()
            offset_w = rand01[6].mul(raw_w + (1 - cutout_w % 2)).floor().long()

            grid_B, grid_h, grid_w = self.get_grids(B, cutout_h, cutout_w, dev)
            grid_h = (grid_h + offset_h).sub_(cutout_h // 2).clamp(min=0, max=raw_h - 1)
            grid_w = (grid_w + offset_w).sub_(cutout_w // 2).clamp(min=0, max=raw_w - 1)
            mask = torch.ones(B, raw_h, raw_w, dtype=BCHW.dtype, device=dev)
            mask[grid_B, grid_h, grid_w] = 0
            BCHW = BCHW.mul(mask.unsqueeze(1))

        return BCHW


###############################################################################
# LPIPS perceptual similarity loss
###############################################################################

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth",
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class VGG16Features(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps: float = 1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim: bool = True):
    return x.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    """Learned perceptual image patch similarity."""

    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 feature sizes
        self.net = VGG16Features(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name: str = "vgg_lpips"):
        ckpt = get_ckpt_path(
            name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        )
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print(f"loaded pretrained LPIPS loss from {ckpt}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_in, y_in = self.scaling_layer(x), self.scaling_layer(y)
        outs0, outs1 = self.net(x_in), self.net(y_in)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


###############################################################################
# DINO-based discriminator (DINODiscriminator)
###############################################################################


class MLPNoDrop(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, fused_if_available=True):
        super().__init__()
        try:
            from flash_attn.ops.fused_dense import fused_mlp_func as _fused_mlp_func
        except Exception:
            _fused_mlp_func = None

        self.fused_mlp_func = (
            _fused_mlp_func if (torch.cuda.is_available() and fused_if_available) else None
        )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation="gelu_approx",
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=0,
                process_group=None,
            )
        else:
            return self.fc2(self.act(self.fc1(x)))


class SelfAttentionNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim=768, num_heads=12, flash_if_available=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        try:
            from flash_attn import flash_attn_qkvpacked_func as _flash_attn_qkvpacked_func
        except Exception:
            _flash_attn_qkvpacked_func = None

        self.using_flash_attn = (
            torch.cuda.is_available() and flash_if_available and _flash_attn_qkvpacked_func is not None
        )
        self._flash_attn_qkvpacked_func = _flash_attn_qkvpacked_func

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash_attn and qkv.dtype != torch.float32:
            oup = self._flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale).view(B, L, C)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            scale = self.scale
            attn = q.mul(scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            oup = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(oup)


class SABlockNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim, num_heads, mlp_ratio, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            flash_if_available=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            fused_if_available=True,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-6):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.float()

        # Reshape batch into groups.
        G = int(np.ceil(x.size(0) / self.virtual_bs))
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool):
    if norm_type == "bn":
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == "sbn":
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=None)
    elif norm_type in {"lbn", "hbn"}:
        # Fallback to SyncBatchNorm without a custom local machine group.
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=None)
    elif norm_type == "gn":
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError

    conv_cls = SpectralConv1d if using_spec_norm else nn.Conv1d
    return nn.Sequential(
        conv_cls(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCHW => BCL => BLC
        return self.norm(x)


class FrozenDINOSmallNoDrop(nn.Module):
    """
    Frozen DINO ViT without any dropout or droppath layers (eval-only),
    based on timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0).
    """

    def __init__(
        self,
        depth=12,
        key_depths=(2, 5, 8, 11),
        norm_eps=1e-6,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.img_size = 224
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_size = patch_size
        self.patch_nums = self.img_size // patch_size

        # x \in [-1, 1]
        m = torch.tensor((0.485, 0.456, 0.406))
        s = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / s).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - m) / s).reshape(1, 3, 1, 1))
        self.crop = RandomCrop(self.img_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim)
        )  # +1: for cls

        self.key_depths = set(d for d in key_depths if d < depth)
        self.blocks = nn.Sequential(
            *[
                SABlockNoDrop(
                    block_idx=i,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_eps=norm_eps,
                )
                for i in range(max(depth, 1 + max(self.key_depths)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # eval mode only
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def inter_pos_embed(self, patch_nums=(14, 14)):
        if patch_nums[0] == self.patch_nums and patch_nums[1] == self.patch_nums:
            return self.pos_embed
        pe_cls, pe_grid = self.pos_embed[:, :1], self.pos_embed[0, 1:]
        pe_grid = pe_grid.reshape(1, self.patch_nums, self.patch_nums, -1).permute(0, 3, 1, 2)
        pe_grid = F.interpolate(
            pe_grid, size=(patch_nums[0], patch_nums[1]), mode="bilinear", align_corners=False
        )
        pe_grid = pe_grid.permute(0, 2, 3, 1).reshape(1, patch_nums[0] * patch_nums[1], -1)
        return torch.cat([pe_cls, pe_grid], dim=1)

    def forward(self, x, grad_ckpt: bool = False):
        with torch.cuda.amp.autocast(enabled=False):
            x = (self.x_scale * x.float()).add_(self.x_shift)
            H, W = x.shape[-2], x.shape[-1]
            if H > self.img_size and W > self.img_size and random.random() <= 0.5:
                x = self.crop(x)
            else:
                x = F.interpolate(
                    x,
                    size=(self.img_size, self.img_size),
                    mode="area" if H > self.img_size else "bicubic",
                )

        x = self.patch_embed(x)

        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x.float()), dim=1)
            x = x + self.pos_embed
            activations = [(x[:, 1:] + x[:, :1]).transpose_(1, 2)]
        for i, b in enumerate(self.blocks):
            if not grad_ckpt:
                x = b(x)
            else:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            if i in self.key_depths:
                activations.append((x[:, 1:].float() + x[:, :1].float()).transpose_(1, 2))
        return activations


class DinoDisc(nn.Module):
    def __init__(
        self,
        dino_ckpt_path="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        device="cuda",
        ks=9,
        depth=12,
        key_depths=(2, 5, 8, 11),
        norm_type="bn",
        using_spec_norm=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        # load state
        state = torch.hub.load_state_dict_from_url(dino_ckpt_path, map_location="cpu")
        for k in sorted(state.keys()):
            if ".attn.qkv.bias" in k:
                bias = state[k]
                C = bias.numel() // 3
                bias[C : 2 * C].zero_()  # zero out k_bias

        key_depths = tuple(d for d in key_depths if d < depth)
        d = FrozenDINOSmallNoDrop(depth=depth, key_depths=key_depths, norm_eps=norm_eps)
        missing, unexpected = d.load_state_dict(state, strict=False)
        missing = [
            m
            for m in missing
            if all(x not in m for x in {"x_scale", "x_shift"})
        ]
        if torch.cuda.is_available():
            assert len(missing) == 0, f"missing keys: {missing}"
            assert len(unexpected) == 0, f"unexpected keys: {unexpected}"

        self.dino_proxy: Tuple[FrozenDINOSmallNoDrop] = (d.to(device=device),)
        dino_C = self.dino_proxy[0].embed_dim

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(
                        dino_C,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        using_spec_norm=using_spec_norm,
                    ),
                    ResidualBlock(
                        make_block(
                            dino_C,
                            kernel_size=ks,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            using_spec_norm=using_spec_norm,
                        )
                    ),
                    (
                        SpectralConv1d
                        if using_spec_norm
                        else nn.Conv1d
                    )(dino_C, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )

    def reinit(
        self,
        dino_ckpt_path="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        device="cuda",
        ks=9,
        depth=12,
        key_depths=(2, 5, 8, 11),
        norm_type="bn",
        using_spec_norm=True,
        norm_eps=1e-6,
    ):
        dino_C = self.dino_proxy[0].embed_dim
        heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(
                        dino_C,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        using_spec_norm=using_spec_norm,
                    ),
                    ResidualBlock(
                        make_block(
                            dino_C,
                            kernel_size=ks,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            using_spec_norm=using_spec_norm,
                        )
                    ),
                    (
                        SpectralConv1d
                        if using_spec_norm
                        else nn.Conv1d
                    )(dino_C, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )

        self.heads.load_state_dict(heads.state_dict())

    def forward(self, x_in_pm1: torch.Tensor, grad_ckpt: bool = False) -> torch.Tensor:
        # x_in_pm1: image tensor normalized to [-1, 1]
        dino_grad_ckpt = grad_ckpt and x_in_pm1.requires_grad
        FrozenDINOSmallNoDrop.forward
        activations: List[torch.Tensor] = self.dino_proxy[0](
            x_in_pm1.float(), grad_ckpt=dino_grad_ckpt
        )
        B = x_in_pm1.shape[0]
        return torch.cat(
            [
                (
                    h(act)
                    if not grad_ckpt
                    else torch.utils.checkpoint.checkpoint(h, act, use_reentrant=False)
                ).view(B, -1)
                for h, act in zip(self.heads, activations)
            ],
            dim=1,
        )


###############################################################################
# GAN loss helpers
###############################################################################


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = loss_real + loss_fake
    return d_loss


def non_saturating_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logits_real), logits_real)
    )
    loss_fake = torch.mean(
        F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake)
    )
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake: torch.Tensor) -> torch.Tensor:
    return -torch.mean(logit_fake)

def vanilla_gen_loss(logit_fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.softplus(-logit_fake))

def non_saturating_gen_loss(logit_fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake), logit_fake)
    )


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0):
    if global_step < threshold:
        weight = value
    return weight


class LeCAM_EMA(object):
    def __init__(self, init: float = 0.0, decay: float = 0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real: torch.Tensor, logits_fake: torch.Tensor):
        self.logits_real_ema = (
            self.logits_real_ema * self.decay
            + torch.mean(logits_real).item() * (1 - self.decay)
        )
        self.logits_fake_ema = (
            self.logits_fake_ema * self.decay
            + torch.mean(logits_fake).item() * (1 - self.decay)
        )


def lecam_reg(real_pred: torch.Tensor, fake_pred: torch.Tensor, lecam_ema: LeCAM_EMA):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


###############################################################################
# Combined GAN loss (generator + discriminator) with perceptual & VQ terms
###############################################################################


class GANLoss(nn.Module):
    """
    Combined loss used for training VQ-style autoencoders with GAN objectives.

    This is functionally similar to the original `VQLoss` but packaged so it can
    live in a single file together with its discriminators and LPIPS.
    """

    def __init__(
        self,
        disc_loss: str = "hinge",
        disc_dim: int = 64,
        disc_type: str = "patchgan",
        image_size: int = 256,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_adaptive_weight: bool = False,
        gen_adv_loss: str = "hinge",
        reconstruction_loss: str = "l2",
        reconstruction_weight: float = 1.0,
        codebook_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        lecam_loss_weight: float = None,
        norm_type: str = "bn",
        aug_prob: float = 1.0,
    ):
        super().__init__()
        # discriminator type & architecture
        assert disc_type in ["patchgan", "stylegan", "dinodisc"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        self.disc_type = disc_type
        if disc_type == "patchgan":
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels,
                image_size=image_size,
            )
        elif disc_type == "dinodisc":
            self.discriminator = DinoDisc(norm_type=norm_type)  # default 224 otherwise crop
            self.daug = DiffAug(prob=aug_prob, cutout=0.2)
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")


        # generator adversarial loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        elif gen_adv_loss == 'vanilla':
            self.gen_adv_loss = vanilla_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss weight (used externally when aggregating)
        self.codebook_weight = codebook_weight

        # LeCAM regularization
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        with fabric.no_backward_sync(last_layer):
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(
        self,
        x,
        x_hat,
        global_step,
        loss='G'
    ):
        """
        Args are kept compatible with the original `VQLoss` for drop-in usage.

        optimizer_idx = 0: generator (autoencoder) update
        optimizer_idx = 1: discriminator update
        """
        # generator update
        if loss == 'G':          
            # discriminator loss (for generator)
            if self.disc_type == "dinodisc":
                if fade_blur_schedule < 1e-6:
                    fade_blur_schedule = 0.0
                logits_fake = self.discriminator(
                    self.daug.aug(x_hat.contiguous(), fade_blur_schedule)
                )
            else:
                logits_fake = self.discriminator(x_hat.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)

            loss = generator_adv_loss
            return (loss,{'gen_G_loss_vanilla':generator_adv_loss.item()})

        # discriminator update
        if loss == 'D':
            if self.disc_type == "dinodisc":
                logits_fake = self.discriminator(
                    self.daug.aug(x_hat.contiguous().detach(), 0.)
                )
                logits_real = self.discriminator(
                    self.daug.aug(x.contiguous().detach(), 0.)
                )
            else:
                logits_fake = self.discriminator(x_hat.contiguous().detach())
                logits_real = self.discriminator(x.contiguous().detach())

         

            if self.lecam_loss_weight is not None:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = self.lecam_loss_weight * lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d = self.disc_loss(logits_real, logits_fake)
                d_adversarial_loss = lecam_loss + non_saturate_d
                
            else:
                d_adversarial_loss = self.disc_weight * self.disc_loss(logits_real, logits_fake)

            logits_real_log = logits_real.detach()
            logits_fake_log = logits_fake.detach()

            return (d_adversarial_loss, { "disc_loss":non_saturate_d.item(), "lecam_loss": lecam_loss.item(), "prob_real":logits_real_log.sigmoid().mean().item(), "prob_fake":logits_fake_log.sigmoid().mean().item()})
