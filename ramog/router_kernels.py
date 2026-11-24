"""Fused one-token router for the softmax -> top-k (-> renormalize) family (Qwen3-MoE's TopKRouter).

transformers' router at batch 1 is a bf16 GEMV through hipBLASLt / cuBLAS (24 us on an MI250 for 128 x 2048), a
float softmax, then ramog.arch.topk_stable's sort; with the glue that is ~60 us per layer, ~3 ms per token on
Qwen3-30B-A3B. Here: one GEMV kernel (logits rounded to the weight dtype, as F.linear returns them) and one
single-program kernel for softmax (float32), the k largest probabilities in descending order with ties by expert
index (topk_stable's order) and the optional renormalization.

    logits, scores, idx = route_softmax_topk(h, weight, k, renorm)     h [1, H]; logits [1, E]; scores [1, k] (dtype
                                                                      of the logits); idx [1, k] int64

and for the sigmoid + correction-bias family without group limits (DeepSeek-V3's noaux_tc with n_group 1, GLM-4.7-Flash:
transformers' TopkRouter computes float32 logits, sigmoid, top-k of sigmoid + e_score_correction_bias through a sort
that inductor lowers to one ~50 us kernel per layer):

    logits, weights, idx = route_sigmoid_topk(h, weight, bias, k, renorm, scale)   logits [1, E] float32; weights
                                                                                    [1, k] float32 (sigmoid of the
                                                                                    chosen, renormalized, x scale)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemv(h_ptr, w_ptr, out_ptr, E, H, stride_w, BLOCK_E: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    ok = rows < E
    acc = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_H):
        cols = k0 + tl.arange(0, BLOCK_H)
        cm = cols < H
        w = tl.load(w_ptr + rows[:, None].to(tl.int64) * stride_w + cols[None, :], mask=ok[:, None] & cm[None, :],
                    other=0.0).to(tl.float32)
        x = tl.load(h_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        acc += w * x[None, :]
    tl.store(out_ptr + rows, tl.sum(acc, axis=1).to(out_ptr.dtype.element_ty), mask=ok)


@triton.jit
def _softmax_topk(logit_ptr, score_ptr, idx_ptr, E, K: tl.constexpr, RENORM: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    ok = e < E
    z = tl.load(logit_ptr + e, mask=ok, other=float("-inf")).to(tl.float32)
    m = tl.max(z, axis=0)
    p = tl.exp(z - m)
    p = tl.where(ok, p, 0.0)
    p = p / tl.sum(p, axis=0)
    work = tl.where(ok, p, -1.0)  # probabilities are >= 0: -1 marks taken / out-of-range slots
    kk = tl.arange(0, K)
    vals = tl.zeros((K,), dtype=tl.float32)
    ids = tl.zeros((K,), dtype=tl.int64)
    for j in tl.static_range(K):
        best = tl.max(work, axis=0)
        sel = tl.min(tl.where(work == best, e, BLOCK), axis=0)  # ties: the lowest expert index
        vals = tl.where(kk == j, best, vals)
        ids = tl.where(kk == j, sel.to(tl.int64), ids)
        work = tl.where(e == sel, -1.0, work)
    if RENORM:
        vals = vals / tl.sum(vals, axis=0)
    tl.store(score_ptr + kk, vals.to(score_ptr.dtype.element_ty))
    tl.store(idx_ptr + kk, ids)


def route_softmax_topk(h: torch.Tensor, weight: torch.Tensor, k: int, renorm: bool):
    e, hdim = weight.shape
    hv = h.reshape(-1)
    logits = torch.empty(1, e, device=h.device, dtype=torch.promote_types(h.dtype, weight.dtype))
    scores = torch.empty(1, k, device=h.device, dtype=logits.dtype)
    idx = torch.empty(1, k, device=h.device, dtype=torch.int64)
    block_e = 4
    _router_gemv[(triton.cdiv(e, block_e),)](hv, weight, logits, e, hdim, weight.stride(0), BLOCK_E=block_e,
                                             BLOCK_H=512, num_warps=4)
    _softmax_topk[(1,)](logits, scores, idx, e, K=k, RENORM=renorm, BLOCK=triton.next_power_of_2(e), num_warps=1)
    return logits, scores, idx


@triton.jit
def _sigmoid_topk(logit_ptr, bias_ptr, w_ptr, idx_ptr, E, scale, K: tl.constexpr, RENORM: tl.constexpr,
                  BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    ok = e < E
    z = tl.load(logit_ptr + e, mask=ok, other=0.0).to(tl.float32)
    sg = 1.0 / (1.0 + tl.exp(-z))
    work = tl.where(ok, sg + tl.load(bias_ptr + e, mask=ok, other=0.0).to(tl.float32), float("-inf"))
    kk = tl.arange(0, K)
    vals = tl.zeros((K,), dtype=tl.float32)
    ids = tl.zeros((K,), dtype=tl.int64)
    for j in tl.static_range(K):
        best = tl.max(work, axis=0)
        sel = tl.min(tl.where(work == best, e, BLOCK), axis=0)  # ties: the lowest expert index
        vals = tl.where(kk == j, tl.sum(tl.where(e == sel, sg, 0.0), axis=0), vals)  # weight: the plain sigmoid
        ids = tl.where(kk == j, sel.to(tl.int64), ids)
        work = tl.where(e == sel, float("-inf"), work)
    if RENORM:
        vals = vals / (tl.sum(vals, axis=0) + 1e-20)
    tl.store(w_ptr + kk, vals * scale)
    tl.store(idx_ptr + kk, ids)


def route_sigmoid_topk(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, k: int, renorm: bool,
                       scale: float):
    e, hdim = weight.shape
    logits = torch.empty(1, e, device=h.device, dtype=torch.float32)
    weights = torch.empty(1, k, device=h.device, dtype=torch.float32)
    idx = torch.empty(1, k, device=h.device, dtype=torch.int64)
    block_e = 4
    _router_gemv[(triton.cdiv(e, block_e),)](h.reshape(-1), weight, logits, e, hdim, weight.stride(0),
                                             BLOCK_E=block_e, BLOCK_H=512, num_warps=4)
    _sigmoid_topk[(1,)](logits, bias, weights, idx, e, float(scale), K=k, RENORM=renorm,
                        BLOCK=triton.next_power_of_2(e), num_warps=1)
    return logits, weights, idx
