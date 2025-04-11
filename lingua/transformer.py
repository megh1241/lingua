# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple
import os
import torch
from torch import nn, autograd
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)
import xformers.ops as xops
import xformers.ops.sp24 as sp24

from lingua import probe
#from lingua import sparse_ops
#from sparse_ops import FP8SparseLinear


import numpy as np

from torch.cuda.amp import custom_fwd, custom_bwd

from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton


flex_attention_comp = torch.compile(flex_attention)

_24_WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS_24", "62000"))
_24_WARMUP = False

_WSPARSIFY1 = False
_WSPARSIFY2 = False
_SP_RATIO = 0.95
_SHUFFLE_ROWS = True
_ACTIVATION_SPARSE = False


def check_24_row( a):
    num_eles = a.size
    for i in range(0, num_eles-4, 4):
        subset_a = a[i:i+4]
        num_zeros = np.count_nonzero(subset_a == 0)
        if num_zeros < 2:
            return False
    return True

def check_24(t):
    is_24 = True
    with torch.no_grad():
        a = t.detach().cpu()
        a = a.to(torch.float32).numpy()
        for i in range(a.shape[0]):
            is_24 = is_24 and check_24_row(a[i,:])
            if not is_24:
                return is_24
    return is_24

class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"
    activation_fn: str = "sqrelu"
    wsparsify1: bool = True
    wsparsify2: bool = True
    activation_sparsify: bool = False
    max_seqlen: int = 1024

def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class TiedLinear(nn.Module):
    def __init__(self, tied_module: nn.Module) -> None:
        super().__init__()
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.tied_module.weight)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )



class SoftThreshold(autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        return weight_sparse * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def get_dense_and_sparse_indices(act: torch.Tensor, sparse_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    assert act.ndim == 2
    sparse_level = (act <= 0).mean(0, dtype=torch.float32)
    sparse_sorted = sparse_level.argsort(descending=True)
    first_dense = int(sparse_ratio * sparse_sorted.shape[0])
    first_dense = (int((first_dense - 1) // 128) + 1) * 128

    idx_d = sparse_sorted[first_dense:]
    idx_sp = sparse_sorted[:first_dense]
    return idx_d, idx_sp


def sp24_dense(x: torch.Tensor, algo="largest_abs") -> torch.Tensor:
    from xformers.ops import sp24

    assert algo in ["largest", "largest_abs"]

    # Also important for evals: we don't want to do sparsity
    # (otherwise would need to pad probably?)
    if _24_WARMUP or not _ACTIVATION_SPARSE:
        return x
    return torch.ops.xformers.sparseNM_dense(x, N=2, M=4, sort_preproc=algo)

def dense_and_sp_mm(a: torch.Tensor, b: torch.Tensor, idx_d: torch.Tensor, idx_sp: torch.Tensor) -> torch.Tensor:
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    assert a.shape == (m, k)
    assert b.shape == (k, n)
    assert idx_d.shape[0] + idx_sp.shape[0] == m, f"{idx_d.shape[0]} + {idx_sp.shape[0]} != {m}"
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    out[idx_d] = a[idx_d] @ b
    out[idx_sp] = sp24_dense(a[idx_sp]) @ b
    return out

class _FFNSRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        """sparse"""
        # linear
        w1 = w1.T
        y1 = x @ w1
        y1 = sp24_dense(y1, "largest")
        y2 = F.relu(y1) ** 2
        y3 = y2.view_as(y2) @ w2
        ctx.save_for_backward(x, w1, w2, y1)
        return y3

    @staticmethod
    def backward(ctx, dy3):
        return _FFNSRelu.backward_sp(ctx, dy3 )

    @staticmethod
    def backward_sp(ctx, dy3):
        """Sparse"""
        x, w1, w2, y1 = ctx.saved_tensors
        y2 = F.relu(y1) ** 2
        idx_d, idx_sp = get_dense_and_sparse_indices(y1, sparse_ratio=_SP_RATIO)
        # linear2
        dy2 = dy3 @ w2.T # dense
        dw2 = dense_and_sp_mm(y2.T, dy3, idx_d, idx_sp)
        # relu
        dy1 = 2 * dy2 * (F.relu(y1))
        # linear1
        dx = sp24_dense(dy1) @ w1.T # sparse
        dw1 = dense_and_sp_mm(dy1.T, x, idx_d, idx_sp).T
        return dx, dw1.T, dw2,  None

    @staticmethod
    def backward_dense(ctx, dy3):
        """Dense"""
        x, w1, w2, y1 = ctx.saved_tensors
        # recompute
        y2 = F.relu(y1) ** 2
        # linear2
        dy2 = dy3 @ w2.T
        if _WSPARSIFY2:
            dw2 = y2.T@MVUE24_approx_triton(dy3)
        else:
            dw2 = y2.T@dy3
        # relu
        dy1 = 2 * dy2 * (F.relu(y1))
        # linear1
        dx = dy1@w1.T
        if _WSPARSIFY1:
            dw1 = x.T@MVUE24_approx_triton(dy1)
        else:
            dw1 = x.T@dy1
        assert dx.shape == x.shape
        assert dw1.shape == w1.shape
        assert dw2.shape == w2.shape
        return dx, dw1, dw2



class FeedForwardSRelu(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
        activation_fn: str = 'sqrelu',
        
        ):
        super().__init__()
        if _WSPARSIFY1:
            self.register_buffer('scale1', torch.tensor(0.))
        if _WSPARSIFY2:
            self.register_buffer('scale2', torch.tensor(0.))
 
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.activation_fn = activation_fn
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        self.w2 = nn.Linear(
                hidden_dim,
                dim,
                bias=False,
        )
  

    def get_sparse_weights_w1(self):
        return SoftThreshold.apply(self.w1.weight, self.scale1)
    
    def get_sparse_weights_w2(self):
        return SoftThreshold.apply(self.w2.weight, self.scale2)

    @torch.no_grad()
    def _init_scale_w(self, w, obj_name = 'scale1'):
        weight = w.cuda()
        weight_temp = weight.detach()
        weight_temp_full = weight_temp.full_tensor()
        weight_sparse, _ = soft_threshold24_triton(weight_temp_full)
        scale = torch.sum(torch.mul(torch.flatten(weight_temp_full),
                    torch.flatten(weight_sparse))) / torch.sum(torch.mul(
                torch.flatten(weight_sparse), torch.flatten(weight_sparse)))
        if obj_name == 'scale1':
            self.scale1.copy_(scale.cpu())
        else:
            self.scale2.copy_(scale.cpu())


    @torch.no_grad()
    def init_scale(self):
        if _WSPARSIFY1:
            self._init_scale_w(self.w1.weight, 'scale1')
        if _WSPARSIFY2:
            self._init_scale_w(self.w2.weight, 'scale2')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.flatten(0, -2)
        _SHUFFLE_ROWS=True
        if _SHUFFLE_ROWS:
            rp = torch.randperm(x.shape[0], device=x.device)
            x = x[rp]
        
        w1 = self.w1.weight
        w2 = self.w2.weight
        if _WSPARSIFY1:
            w1 = self.get_sparse_weights_w1()
        if _WSPARSIFY2:
            w2 = self.get_sparse_weights_w2()
        out = out_permed = _FFNSRelu.apply(x, w1, w2.T)
        
        if _SHUFFLE_ROWS:
            out = torch.empty_like(out_permed)
            out[rp] = out_permed
       
        return out.reshape(orig_shape)
    
  
    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
 
        nn.init.trunc_normal_(
                self.w1.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )

        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
        activation_fn: str = 'sqrelu',
        ):
        super().__init__()

        if activation_fn != 'sqrelu':
            hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.activation_fn = activation_fn
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        if self.activation_fn != 'sqrelu':
            self.w3 = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )
        self.w2 = nn.Linear(
                hidden_dim,
                dim,
                bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        if self.activation_fn == 'sqrelu':
            y1 = self.w1(x.view_as(x))
            y2 = F.relu(y1) ** 2
            return self.w2(y2.view_as(y2))
        else:
            x1 = self.w1(x.view_as(x))
            x3 = self.w3(x.view_as(x))
            return self.w2(F.silu(x1) * x3)
        
    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        if self.activation_fn == 'sqrelu':
            nn.init.trunc_normal_(
                self.w1.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        else:
            for w in [self.w1, self.w3]:
                nn.init.trunc_normal_(
                    w.weight,
                    mean=0.0,
                    std=in_init_std,
                    a=-3 * in_init_std,
                    b=3 * in_init_std,
                )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        #Note: only set to True to do a sanity check.
        #FeedForwardSRelu with wsparsify1, wsparsify2 and
        #activation_sparsity set to false is equivalent to baseline.
        baseline = False 
        if baseline:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                activation_fn=args.activation_fn,
            )
        else:
            self.feed_forward = FeedForwardSRelu(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                activation_fn=args.activation_fn,
            )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa"
    ) -> torch.Tensor:

        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))

        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()
        
        if _WSPARSIFY2 or _WSPARSIFY1:
            self.feed_forward.init_scale()
    

class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )


        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))


    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask,
                    attn_impl=attn_impl)
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()


    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]
            layer.init_weights(self.init_base_std, factor)
