import math
from math import pi

import torch
import torch.nn as nn
from torch import einsum, broadcast_tensors, Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast
from torch import nn
from einops import rearrange, repeat
import einops
import numpy as np
import os
import copy

# Well Inited Linear
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    if mode == 'uniform': return torch.rand() * 2 - 1
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_uniform', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class AdaRMSNorm(nn.Module):
    def __init__(self,dim, cond_dim=None, elementwise_affine=True, cond_based_affine=True, centering=False, eps=None):
        super(AdaRMSNorm, self).__init__()        
        self.dim = dim
        self.eps = eps
        self.cond_based_affine = cond_based_affine      
        self.elementwise_affine = elementwise_affine
        self.centering = centering  
        assert not(cond_dim is None and cond_based_affine), 'cond_dim must be provided if cond_based_affine is True'
        if elementwise_affine:
            if cond_based_affine and cond_dim is not None:
                self.affine = Linear(cond_dim, dim, init_weight=1e-5)
            else:
                self.weight = nn.Parameter(torch.ones(self.dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x, cond_emb=None):   
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        rms = torch.rsqrt(torch.square(x).mean(dim=-1, keepdim=True) + eps)
        out = ((x - x.mean(dim=-1, keepdim=True)) * rms) if self.centering else (x * rms)
        if self.elementwise_affine:
            if self.cond_based_affine:
                out = out.mul(1. + self.affine(cond_emb).unsqueeze(1))
            else:
                out = out.mul(self.weight)
        return out

class AdaLN(nn.Module):
    def __init__(self,dim, cond_dim=None, elementwise_affine=True, cond_based_affine=True, bias=False):
        super(AdaLN, self).__init__()        
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine and not cond_based_affine)
        self.cond_based_affine = cond_based_affine
        self.bias = bias
        assert not(cond_dim is None and cond_based_affine), 'cond_dim must be provided if cond_based_affine is True'
        self.affine = Linear(cond_dim,  2 * dim if bias else dim, init_weight=1e-5) if cond_based_affine and cond_dim is not None else None      
    def forward(self, x, cond_emb=None):  
        x = self.norm(x)
        if self.cond_based_affine:
            if self.bias:
                shift, scale = self.affine(cond_emb).unsqueeze(1).chunk(2, dim=-1)
                x = x.mul(1. + scale).add_(shift)
            else:
                scale = self.affine(cond_emb).unsqueeze(1)
                x = x.mul(1. + scale)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(frequency_embedding_size, hidden_size, init_weight=1e-5),
            nn.SiLU(),
            Linear(hidden_size, hidden_size, init_weight=1e-5),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# ROPE
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype
    with torch.amp.autocast('cuda',enabled=False):
        freqs = freqs.float()
        t = t.float()
        if not exists(freqs_seq_dim):
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0

        if t.ndim == 3 or exists(freqs_seq_dim):
            seq_len = t.shape[seq_dim]
            freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        # Split t into three parts: left, middle (to be transformed), and right
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]

        # Apply rotary embeddings without modifying t in place    
        t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
            
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs= None,
        freqs_for= 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = True,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq
        
   
        # dummy for device
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0



    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return self.apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None, offset = 0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)
        scale = self.get_scale(seq, seq_len = seq_len, offset = offset).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len= None,
        offset = 0
    ):

        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    def forward(
        self,
        t,
        seq_len = None,
        offset = 0
    ):
        dtype = t.dtype
        with torch.amp.autocast('cuda',enabled=False):
            t = t.float()
            freqs = self.freqs.float()

            should_cache = (
                self.cache_if_possible and
                not self.learned_freq and
                exists(seq_len) and
                self.freqs_for != 'pixel' and
                (offset + seq_len) <= self.cache_max_seq_len
            )

            if (
                should_cache and \
                exists(self.cached_freqs) and \
                (offset + seq_len) <= self.cached_freqs_seq_len
            ):
                return self.cached_freqs[offset:(offset + seq_len)].detach()


            freqs = einsum('..., f -> ... f', t, freqs)
            freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

            if should_cache and offset == 0:
                self.cached_freqs[:seq_len] = freqs.detach()
                self.cached_freqs_seq_len = seq_len

        return freqs.type(dtype)

# DropPath & FFN
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., ffn_type='geglu'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.ffn_type = ffn_type
        if self.ffn_type == 'geglu':
            self.fc1 = Linear(in_features, 2*hidden_features)
            self.act = nn.GELU(approximate='tanh')
            self.fc2 = Linear(hidden_features, out_features)
        elif self.ffn_type == 'ffn':
            self.fc1 = Linear(in_features, hidden_features)
            self.act = nn.GELU(approximate='tanh')
            self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.ffn_type == 'geglu':
            gate, value = self.fc1(x).chunk(2, dim=-1)
            return self.drop(self.fc2(self.act(value)*gate))
        elif self.ffn_type == 'ffn':
            return self.drop(self.fc2(self.act(self.fc1(x))))
    

class Attention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, 
        rope=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        self.rope = RotaryEmbedding(embed_dim//num_heads) if rope else None
        
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.to_q = Linear(embed_dim, embed_dim, bias=False)
        self.to_kv = Linear(embed_dim, embed_dim * 2, bias=False)

        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop    
    
    def forward(self, x, context_emb=None, causal=False, attn_bias=None, cache_kv=False, past_kvs=None):
        B, L, C = x.shape

        q = self.to_q(x) 
        q = einops.rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        
        kv = self.to_kv(x) if context_emb is None else self.to_kv(context_emb)
        kv = einops.rearrange(kv, 'b l (h d) -> b l h d', h=self.num_heads)
        k, v = kv.chunk(2, dim=-1)

        if self.rope is not None:
            past_len = 0
            if past_kvs is not None:
                past_len = past_kvs[0].shape[2]
            q, k = self.rope.rotate_queries_and_keys(q, k, offset=past_len)
        
        q = q.permute(0, 2, 1, 3)  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3)  # [B, H, L, D]
        v = v.permute(0, 2, 1, 3)  # [B, H, L, D]

        if past_kvs is not None:
            past_keys, past_values = past_kvs
            k = torch.cat((past_keys, k), dim=2)
            v = torch.cat((past_values, v), dim=2)
    
        dropout_p = self.attn_drop if self.training else 0.0
        oup = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            scale=self.scale, 
            is_causal=causal and past_kvs is None,
            dropout_p=dropout_p,
            attn_mask=attn_bias
        ).transpose(1, 2).reshape(B, L, -1)

        out = self.proj_drop(self.proj(oup))
        if not cache_kv:
            return out
        else:
            return out, (k,v)

class TransformerLayer(nn.Module):
    def __init__(
        self, block_idx, embed_dim, cond_dim,
        num_heads, mlp_ratio=8/3, drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False, cross_attn=False, rope=False,
        ffn_type='geglu', norm_layer=AdaRMSNorm
    ):
        super(TransformerLayer, self).__init__()
        self.block_idx = block_idx
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adaln_1 = norm_layer(embed_dim, cond_dim)
        self.attn1 = Attention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, rope=rope)
        self.gate1 = Linear(cond_dim, embed_dim, init_weight=1e-5)

        if cross_attn:
            self.has_cross_attn = True
            self.adaln_2 = norm_layer(embed_dim, cond_dim)
            self.adaln_ctx = norm_layer(embed_dim, cond_dim)

            self.attn2 = Attention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, rope=rope)
            self.gate2 = Linear(cond_dim, embed_dim, init_weight=1e-5)
        else:
            self.has_cross_attn = False

        self.adaln_mlp = norm_layer(embed_dim, cond_dim)
        mlp_ratio = eval(mlp_ratio) if isinstance(mlp_ratio, str) else mlp_ratio
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, ffn_type=ffn_type)
        self.gate_mlp = Linear(cond_dim, embed_dim, init_weight=1e-5)
        
    def forward(self, x,context_emb=None,  cond_emb=None, causal=False, attn_mask=None, cache_kv=False, past_kvs=None):
        # If we are in "concat" mode, `context_emb` is None and masking should apply to self-attention.
        # If we are in "cross_attn" mode, masking should apply to cross-attention (see below).
        self_attn_mask = attn_mask if (attn_mask is not None and context_emb is None) else None

        attn_out1 = self.attn1(
            self.adaln_1(x, cond_emb),
            context_emb=None,
            causal=causal,
            attn_bias=self_attn_mask,
            cache_kv=cache_kv,
            past_kvs=past_kvs,
        )
        if cache_kv:
            attn_out1, new_kvs = attn_out1[0], attn_out1[1]
        
        x = x + self.gate1(cond_emb).unsqueeze(1) * self.drop_path(attn_out1)
        if context_emb is not None and self.has_cross_attn:
            x = x + self.gate2(cond_emb).unsqueeze(1) * self.drop_path(self.attn2(self.adaln_2(x, cond_emb), context_emb=self.adaln_ctx(context_emb,cond_emb), causal=False, attn_bias=attn_mask))
        x = x + self.gate_mlp(cond_emb).unsqueeze(1) * self.drop_path(self.ffn(self.adaln_mlp(x, cond_emb)))
    
        if cache_kv:
            return x, new_kvs
        else:
            return x

class Transformer(nn.Module):
    def __init__(
        self, 
        layer_num, 
        input_dim, dim, output_dim, max_seq_len, heads,
        cond_input_dim, cond_dim, 
        context_type='none', ctx_input_dim=None, ctx_max_seq_len=None, 
        emb_dropout=0., drop=0., 
        rope=False, abs_pos=False, 
        l2_norm=False,  causal=False, 
        zero_out=False, 
        query_type='learnable', # 'learnable', 'noise', 'none'
        out_act=True,
        ffn_type='geglu', # ffn or geglu
        norm_layer='RMSNorm', # LayerNorm or RMSNorm
        mlp_ratio=8/3, **params
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        

       
        self.query_type = query_type
            
        self.cond_input_dim = cond_input_dim
        self.dim = dim
        self.causal = causal
        self.cross_attn = context_type=='cross_attn'
        norm_layer = AdaRMSNorm if norm_layer=="RMSNorm" else AdaLN
        self.layers = nn.ModuleList([
            TransformerLayer(block_idx=i, embed_dim=dim, 
            cond_dim=cond_dim, num_heads=heads, mlp_ratio=mlp_ratio, 
            drop=drop, attn_drop=drop, attn_l2_norm=l2_norm, rope=rope, cross_attn=self.cross_attn, 
            ffn_type=ffn_type, norm_layer=norm_layer)
            for i in range(layer_num)
        ])

        out_weight = 1e-5 if zero_out else 1
        scale = np.sqrt(dim)
        self.adaln_out = norm_layer(dim, cond_dim)

        self.out_activate = nn.GELU(approximate='tanh') if out_act else None
        self.out = Linear(dim, output_dim,init_weight=out_weight)

        if self.query_type!='noise':
            self.cond_proj = nn.Sequential(Linear(cond_input_dim, cond_dim),norm_layer(cond_dim, cond_based_affine=False), nn.GELU(approximate='tanh'))
        else:
            self.cond_proj = Linear(cond_input_dim, cond_dim)
            self.cond_norm_act = nn.Sequential(norm_layer(cond_dim, cond_based_affine=False), nn.GELU(approximate='tanh'))
        self.max_seq_len = max_seq_len
        self.context_type = context_type # "concat, cross_attn, none"
        if abs_pos:
            if context_type == 'none':
                self.abs_pos = nn.Embedding(max_seq_len, dim)
            elif context_type == 'concat':
                ctx_max_seq_len = default(ctx_max_seq_len, max_seq_len)
                self.abs_pos = nn.Embedding(max_seq_len + ctx_max_seq_len, dim)
            elif context_type == 'cross_attn':
                ctx_max_seq_len = default(ctx_max_seq_len, max_seq_len)
                self.abs_pos = nn.Embedding(max_seq_len, dim)
                self.ctx_pos = nn.Embedding(ctx_max_seq_len, dim)
            else:
                raise ValueError(f"Invalid context_type: {context_type}")
        else:
            self.abs_pos = None

        if context_type == 'concat':
            self.input_layer = Linear(input_dim, dim)
            self.learnable_query_param = nn.Parameter(torch.zeros(1, max_seq_len, input_dim).normal_(0, 1)) if self.query_type == 'learnable' else None
            ctx_input_dim = default(ctx_input_dim, input_dim)
            self.ctx_input_dim = ctx_input_dim
            self.ctx_input_layer =  Linear(ctx_input_dim, dim)
        elif context_type == 'cross_attn':
            self.input_layer = Linear(input_dim, dim)
            ctx_input_dim = default(ctx_input_dim, input_dim)
            self.ctx_input_dim = ctx_input_dim
            self.learnable_query_param = nn.Parameter(torch.zeros(1, max_seq_len, input_dim).normal_(0, 1)) if self.query_type == 'learnable' else None
            self.ctx_input_layer =  Linear(ctx_input_dim, dim)
        else:
            self.input_layer = Linear(input_dim, dim)
            
        if self.query_type == 'noise':
            self.t_embedder = TimestepEmbedder(cond_input_dim)
            self.t_proj = Linear(cond_input_dim,cond_dim)

    def forward(self, x, condition, attn_mask=None, cache_kv=False, past_kvs=None, query_tokens=None, t=None):
        
        # Inject time embedding if using noise query
        if self.query_type == 'noise':
            cond_emb = self.cond_proj(condition)
            if t is not None:
                t_emb = self.t_embedder(t)
                t_emb = self.t_proj(t_emb)
                cond_emb = cond_emb + t_emb
            cond_emb = self.cond_norm_act(cond_emb)
        else:
            cond_emb = self.cond_proj(condition)
        if self.context_type == 'none':
            x, context_emb = self.input_layer(x.contiguous()), None
        elif self.context_type == 'concat':
            B = x.size(0)
            if query_tokens is None:
                if self.query_type == 'learnable':
                    query_tokens = self.learnable_query_param.expand(B, -1, -1)
                else:
                    # Should be provided for noise query, or error if not available
                    raise ValueError(f"query_tokens is required when query_type is '{self.query_type}'.")

            query_tokens = self.input_layer(query_tokens.contiguous())
            x = self.ctx_input_layer(x.contiguous())
            x =  torch.cat((query_tokens,x), dim=1)
            x, context_emb = x, None
        else:
            # cross_attn: x is context (keys/values), query_tokens are queries
            B = x.size(0)
            if query_tokens is None:
                if self.query_type == 'learnable':
                    query_tokens = self.learnable_query_param.expand(B, -1, -1)
                else:
                    raise ValueError(f"query_tokens is required when query_type is '{self.query_type}'.")

            x, context_emb = self.input_layer(query_tokens.contiguous()), self.ctx_input_layer(x.contiguous())
            
        if self.abs_pos is not None:
            if self.context_type == 'none':
                pos_ids = torch.arange(x.size(1), device=x.device)
                if past_kvs is not None:
                    pos_ids = pos_ids + past_kvs[0][0].shape[2]
                x = x + self.abs_pos(pos_ids)
            elif self.context_type == 'concat':
                pos_ids = torch.arange(x.size(1), device=x.device)
                x = x + self.abs_pos(pos_ids)
            elif self.context_type == 'cross_attn':
                pos_ids = torch.arange(x.size(1), device=x.device)
                x = x + self.abs_pos(pos_ids)
                if context_emb is not None and self.ctx_pos is not None:
                    context_emb = context_emb + self.ctx_pos(torch.arange(context_emb.size(1), device=context_emb.device))

        kv_caches = [] 
        for i,layer in enumerate(self.layers):
            layer_output = layer(x, context_emb, cond_emb, self.causal, attn_mask, cache_kv, past_kvs[i] if past_kvs is not None else None)
            if cache_kv:
                x, new_kvs = layer_output[0], layer_output[1]
                kv_caches.append(new_kvs)
            else:
                x = layer_output
        if self.context_type == 'concat':
            x = x[:, :self.max_seq_len]
        out = self.adaln_out(x, cond_emb)
        if self.out_activate is not None:   
            out = self.out_activate(out)
        out = self.out(out)
        
        if cache_kv:
            return out, kv_caches
        else:
            return out


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.enc = Transformer(**config["Encoder"])
        if config["vq_type"] == "HGQ":
            self.quantizer = HGQ(**config["Quantizer"])
        elif config["vq_type"] == "VQ":
            self.quantizer = VQ(**config["Quantizer"])
        else:
            raise ValueError(f"Invalid quantizer type: {config['vq_type']}")
    def forward(self, x, labels, training=True):
        h = self.enc(x, labels)
        quant, idx, one_hot, vqloss = self.quantizer(h, labels, training=training)
        return quant, idx, one_hot, vqloss
    
    
class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dec = Transformer(**config["Decoder"])
    
    
    def forward(self, quant, labels, attn_mask=None, query_tokens=None, t=None):
        x_hat = self.dec(quant, labels, attn_mask=attn_mask, query_tokens=query_tokens, t=t)
        return x_hat


class ARModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.bos = nn.Parameter(torch.zeros(1, 1, config["ARModel"]["input_dim"]).normal_(0,1))
        self.semantic_emb = nn.Embedding(config["Quantizer"]["codebook_size"], config["ARModel"]["input_dim"])
        self.ar_model = Transformer(**config["ARModel"])

        self.temperature = config["ARModel"]["temperature"]
        self.max_length = config["ARModel"]["max_seq_len"]

       
    def forward(self,idx,labels, temperature=None):
        bos = self.bos.expand(idx.shape[0], -1, -1)
        shift_input = torch.cat([bos, self.semantic_emb(idx)[:,:-1]],dim=1)
        logits = self.ar_model(shift_input, labels)
        
        if temperature is not None:
            logits = logits / temperature
        elif self.temperature is not None:
            logits = logits / self.temperature
        return logits

    @torch.no_grad()
    @torch._dynamo.disable
    def sampling(self, bz, class_label=None, temperature=1.0,  topK=None, topP=None, cfg=16.0, cfg_schedule='cosine', cfg_power=2.75, cache_kv=False):
        cfg = 0. if class_label is None else cfg
        if cfg>0. :
            num_classes = class_label.shape[1]
            uncond_class_label = F.one_hot(torch.full((bz,), num_classes - 1, device=class_label.device, dtype=torch.long), num_classes=num_classes).float()
        quant_input = self.bos.expand(bz, -1, -1)
        quant_output = []
        past_kvs, uncond_past_kvs = None, None
        for step in range(self.max_length):
            ar_out = self.ar_model(quant_input, class_label, cache_kv=cache_kv, past_kvs=past_kvs)
            if cache_kv:
                logits, past_kvs = ar_out
            else:
                logits = ar_out
            logits = logits[:,-1]
            if cfg>0.:
                if cfg_schedule == 'constant':
                    cfg_scale = cfg
                elif cfg_schedule == 'linear':
                    cfg_scale = 1.0 * (1-step/self.max_length) + cfg * (step/self.max_length)
                elif cfg_schedule == 'cosine':
                    cfg_scale = (1 - math.cos(
                        ((step / self.max_length) ** cfg_power) * math.pi)) * 1/2
                    cfg_scale = (cfg - 1) * cfg_scale + 1
                else:
                    raise ValueError(f"Invalid cfg_schedule: {cfg_schedule}")
                uncond_ar_out = self.ar_model(quant_input, uncond_class_label, cache_kv=cache_kv, past_kvs=uncond_past_kvs)
                if cache_kv:
                    uncond_logits, uncond_past_kvs = uncond_ar_out
                else:
                    uncond_logits = uncond_ar_out
                uncond_logits = uncond_logits[:,-1]
                logits = cfg_scale * logits + (1 - cfg_scale) * uncond_logits
            logits = logits / temperature

            if topK is not None and topK>0.:
                top_logits, top_indices = logits.topk(topK, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(dim=-1, index=top_indices, src=top_logits)
            if topP is not None and 0.<topP<1.:
                sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                probs_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = probs_sum > topP
                mask[..., 1:] = mask[...,:-1].clone()
                mask[...,0] = False
                sorted_logits[mask] = float('-inf')
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            next_idx = torch.multinomial(F.softmax(logits, dim=-1), 1)
            quant_output.append(next_idx)
            if not cache_kv:
                quant_input = torch.cat((quant_input, self.semantic_emb(next_idx)), dim=1)
            else:
                quant_input = self.semantic_emb(next_idx)
        quant_output = torch.cat(quant_output, dim=1)
        return quant_output



# deterministic Hard-Gumbel Softmax Quantizer   
class HGQ(nn.Module):
    def __init__(self, codebook_size, dim, z_dim, cond_dim, cond_based_affine=False, norm_layer='RMSNorm',temperature=1.0, **params):
        super().__init__()
        self.codebook_size = codebook_size  # codebook size (K)
        self.dim = dim  # embedding dimension (D)
        self.z_dim = z_dim
        # Initialize embedding table
        self.proj_in = Linear(dim, z_dim)
        self.codebook = nn.Embedding(codebook_size, z_dim)
        self.codebook.weight.data.uniform_(-1e-4, 1e-4)
        self.norm_layer = norm_layer
        if norm_layer == 'RMSNorm':
            self.z_norm =  AdaRMSNorm(z_dim, cond_dim, elementwise_affine=True, cond_based_affine=cond_based_affine)
            self.code_norm =  AdaRMSNorm(z_dim, cond_dim, elementwise_affine=True, cond_based_affine=cond_based_affine)
        elif norm_layer == 'LN':
            self.z_norm =  AdaLN(z_dim, cond_dim, elementwise_affine=True, cond_based_affine=cond_based_affine)
            self.code_norm =  AdaLN(z_dim, cond_dim, elementwise_affine=True, cond_based_affine=cond_based_affine)
        
        self.out_proj = Linear(z_dim, dim, bias=False)
        self.total_cnt = 0
        self.book_cnt = torch.zeros(codebook_size)
        self.temperature=temperature
        self.float()

    def forward(self, x, labels=None, training=True, log_usage=False, indices=None):
        B, L, D = x.shape
        x = x.float()
        z = self.proj_in(x)

        if self.norm_layer != 'none':
            if self.norm_layer=='l2':
                z= F.normalize(z, dim=-1)
                codes= F.normalize(self.codebook.weight, dim=-1)
            elif self.norm_layer=='RMSNorm':
                z= self.z_norm(z, labels)
                codes= self.code_norm(self.codebook.weight, labels)
            elif self.norm_layer=='LN':
                z= self.z_norm(z, labels)
                codes= self.code_norm(self.codebook.weight, labels)
            else:
                raise ValueError(f"Invalid norm layer: {self.norm_layer}")

        logits = torch.einsum('bld,nd->bln', z, codes)
        prob = F.softmax(logits/self.temperature, dim=-1)

        indices = torch.argmax(prob, dim=-1) if indices is None else indices
        one_hot_ng = F.one_hot(indices, prob.shape[-1]).view(B, L, -1).to(z.device).to(z.dtype)
        one_hot = prob + (one_hot_ng - prob).detach()
 
        quant = torch.einsum('bln,nd->bld', one_hot, codes)
        loss = 0.
        quant_out = self.out_proj(quant)
       
        return quant_out, indices, one_hot, loss
        
    def get_codes_w_onehot(self, one_hot, labels=None, **params):
        codes = self.codebook.weight
        quant = torch.einsum('bln,nd->bld', one_hot, codes)

        if self.norm_layer != 'none':
            if self.norm_layer=='l2':
                quant= F.normalize(quant, dim=-1)
            elif self.norm_layer=='RMSNorm':
                quant= self.code_norm(quant, labels)
            elif self.norm_layer=='LN':
                quant= self.code_norm(quant, labels)
            else:
                raise ValueError(f"Invalid norm layer: {self.norm_layer}")
        return self.out_proj(quant)
    def get_codes_w_indices(self, indices, labels=None, **params):
        codes = self.codebook(indices)
        if self.norm_layer != 'none':
            if self.norm_layer=='l2':
                codes= F.normalize(codes, dim=-1)
            elif self.norm_layer=='RMSNorm':
                codes= self.code_norm(codes, labels)
            elif self.norm_layer=='LN':
                codes= self.code_norm(codes, labels)
            else:
                raise ValueError(f"Invalid norm layer: {self.norm_layer}")
        return self.out_proj(codes)


class VQ(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 dim: int = 256,
                 z_dim: int = 256,
                 beta: float = 0.25,
                 use_norm: bool = False,
                 temperature=1.0, **params
                 ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.z_dim = z_dim
        self.beta = beta
        self.total_cnt = 0
        self.book_cnt = torch.zeros(codebook_size)
        self.proj_in = Linear(dim, z_dim)
        self.embedding = torch.nn.Embedding(codebook_size, z_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_norm = use_norm
        self.proj_out = Linear(z_dim, dim)
        self.register_buffer('ema_vocab_hit_SV', torch.full((1, self.codebook_size), fill_value=0.0))
        self.record_hit = 0
        self.temperature=temperature
        self.float()


    def forward(self, x: torch.Tensor, labels=None, training=True):
        x=x.float()
        z = self.proj_in(x)

        if self.use_norm:
            z = F.normalize(z, dim=-1)
            embedding = F.normalize(self.embedding.weight, dim=-1)
            dist = -1 * torch.einsum('bld,nd->bln', z, embedding)
        else:
            embedding = self.embedding.weight

            dist = torch.sum(z**2, dim=-1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bld,nd->bln', z, embedding)

        indices = torch.argmin(dist, dim=-1) # num_ele
        ste_prob = (-dist/self.temperature).softmax(dim=-1)
        one_hot_ng = F.one_hot(indices, dist.shape[-1]).view(x.shape[0], x.shape[1], -1).to(z.device).to(z.dtype) 
        one_hot = ste_prob + (one_hot_ng - ste_prob).detach()

        quant = embedding[indices]

        loss = self.beta * torch.mean((quant.detach() - z) **2) +  torch.mean((quant - z.detach()) **2) if training else 0.


        # preserve gradients
        quant = z + (quant - z).detach()
        quant = self.proj_out(quant)

        return quant, indices, one_hot, loss
    def get_codes_w_onehot(self, one_hot, labels=None, **params):
        codes = self.embedding.weight
        quant = torch.einsum('bln,nd->bld', one_hot, codes)
        quant = F.normalize(quant, dim=-1) if self.use_norm else quant
        return self.proj_out(quant)
    def get_codes_w_indices(self, indices):
        quant = self.embedding(indices)
        quant = F.normalize(quant, dim=-1) if self.use_norm else quant
        return self.proj_out(quant)
