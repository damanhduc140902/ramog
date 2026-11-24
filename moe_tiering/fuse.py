"""Decode-step fusion: torch.compile each decoder layer for the one-token decode shape (design: agents/speed).

    compile_decode(model)      before the decode session captures its graph (session.GreedyDecoder / Generator)

transformers' decoder layers run ~100 small kernels per token (norms in float32, casts, rotary, router top-k, residual
adds); at batch 1 their fixed cost, not the weights, set the step time (Qwen3-30B-A3B on an MI250 GCD: 36.8 ms per
token, 39 ms of kernels). Inductor fuses them; the engine's own kernels (TieredExperts, QuantLinear, KLinear,
QEmbedding) stay outside the compiled graphs (torch.compiler.disable), so their tuned launches and the cache manager
are unchanged. Prefill (more than one token) runs the original forward, so prompts of any length never recompile.
Measured (agents/speed/compile_probe.py, Qwen3-30B TQ2.57 resident): 36.8 -> 25.7 ms per token, greedy tokens equal,
KL to the uncompiled engine 0.0015 mean over 33 teacher-forced steps.
"""

from __future__ import annotations

import torch

from . import arch


def _engine_kernels():
    from .experts import TieredExperts
    from .klinear import KLinear
    from .qembed import QEmbedding
    from .qlinear import QuantLinear

    return (TieredExperts, QuantLinear, KLinear, QEmbedding)


def compile_decode(model, mode: str = "default") -> int:
    """Compile every decoder layer's one-token forward in place; returns the number of layers. Idempotent."""
    if getattr(model, "_moe_tiering_compiled", False):
        return 0
    kinds = _engine_kernels()
    for m in model.modules():
        if isinstance(m, kinds) and not getattr(m, "_compile_disabled", False):
            m.forward = torch.compiler.disable(m.forward)
            m._compile_disabled = True
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 512)
    torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 4096)
    torch._dynamo.config.allow_unspec_int_on_nn_module = True  # layer_idx: one graph for every layer
    layers = arch.decoder_layers(model)
    for li, layer in enumerate(layers):
        orig = layer.forward
        comp = torch.compile(orig, dynamic=False, mode=mode)
        state = {"ok": True}

        def fwd(*args, _o=orig, _c=comp, _s=state, _i=li, **kw):
            hs = args[0] if args else kw["hidden_states"]
            if hs.shape[-2] != 1 or not _s["ok"]:
                return _o(*args, **kw)
            try:
                return _c(*args, **kw)
            except Exception as e:  # noqa: BLE001 - compiling is lazy: an architecture it cannot handle keeps eager
                if torch.cuda.is_current_stream_capturing():
                    raise
                _s["ok"] = False
                print(f"moe-tiering: layer {_i} runs uncompiled ({type(e).__name__}: {str(e)[:160]})", flush=True)
                return _o(*args, **kw)

        layer.forward = fwd
    model._moe_tiering_compiled = True
    return len(layers)


def _decode_attention(orig):
    """transformers' sdpa attention with a matmul path for one query token: [Hkv, G, D] x [Hkv, L, D] scores in
    float32, the mask applied, softmax, x V; inside a compiled layer inductor fuses the glue. The original handles
    everything else (prefill, attention sinks, logit softcapping, dropout, position bias)."""

    def attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
        cfg = getattr(module, "config", None)
        if getattr(module, "_mla_absorb", False):  # key / value are MLA latents (install_mla_absorb)
            if query.shape[2] == 1 and attention_mask is not None and not dropout:
                return _mla_decode(module, query, key, value, attention_mask, scaling)
            key, value = module._mla_expand(key, value)
        if (query.shape[2] != 1 or dropout or kw.get("s_aux") is not None or kw.get("sinks") is not None
                or kw.get("position_bias") is not None or kw.get("cache") is not None
                or getattr(cfg, "attn_logit_softcapping", None) or attention_mask is None
                or query.shape[1] % key.shape[1]):
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
        b, hq, _, d = query.shape
        hkv, n = key.shape[1], key.shape[2]
        s = scaling if scaling is not None else d ** -0.5
        # scores as products and sums (inductor fuses them into one Triton reduction; a one-row matmul through the
        # vendor BLAS costs ~20 us on an MI250); p x V stays a matmul (as a reduction over L it was 3x slower)
        q = query.reshape(b, hkv, hq // hkv, 1, d).float()
        sc = (q * key.float()[:, :, None]).sum(-1) * s  # [B, Hkv, G, L]
        m = attention_mask[:, :, -1:, :n]
        sc = sc.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else sc + m.float()
        o = torch.matmul(torch.softmax(sc, dim=-1), value.float())  # [B, Hkv, G, D]
        return o.to(query.dtype).reshape(b, hq, 1, d).transpose(1, 2).contiguous(), None

    attn._moe_tiering = True
    return attn


def _mla_decode(module, query, kv, kr, attention_mask, scaling):
    """MLA attention of one query token straight over the latent cache (weight absorption): with kv_b_proj's rows
    split per head into W_UK [Dn, R] and W_UV [Dv, R], q_nope . (W_UK c_t) = (W_UK^T q_nope) . c_t and
    sum_t p_t W_UV c_t = W_UV (sum_t p_t c_t), so a step reads the latent cache once instead of expanding every
    cached position through kv_b_proj (transformers 5.19 caches the latent and expands all of it per token: a
    cache-length GEMM per layer). Float32 throughout, rounded to the query's dtype at the end.
    query [B, H, 1, Dn + P]; kv [B, 1, T, R] (kv_a_layernorm'd latent); kr [B, 1, T, P] (roped shared key)."""
    dn, dv = module.qk_nope_head_dim, module.v_head_dim
    b, h = query.shape[:2]
    r, t = kv.shape[-1], kv.shape[2]
    w = module.kv_b_proj.weight.view(h, dn + dv, r)
    q = query[:, :, 0].float()                                          # [B, H, Dn + P]
    q_lat = (q[..., :dn, None] * w[None, :, :dn].float()).sum(-2)      # [B, H, R]
    lat = kv[:, 0].float()                                              # [B, T, R]
    sc = (q_lat[:, :, None] * lat[:, None]).sum(-1)                     # [B, H, T]
    sc = sc + (q[:, :, None, dn:] * kr[:, 0, None].float()).sum(-1)
    s = scaling if scaling is not None else query.shape[-1] ** -0.5
    sc = sc[:, :, None] * s                                             # [B, H, 1, T]
    m = attention_mask[:, :, -1:, :t]
    sc = sc.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else sc + m.float()
    o_lat = torch.matmul(torch.softmax(sc, dim=-1)[:, :, 0], lat)      # [B, H, R]
    o = (o_lat[:, :, None, :] * w[None, :, dn:].float()).sum(-1)       # [B, H, Dv]
    return o.to(query.dtype)[:, None].contiguous(), None                # [B, 1, H, Dv]


def _latent_expand_kv(self, kv_nope, k_rot):
    if getattr(self, "_mla_absorb", False):
        return kv_nope, k_rot  # the sdpa wrapper absorbs (one token) or expands (_mla_expand)
    return self._mla_expand(kv_nope, k_rot)


def install_mla_absorb(model) -> int:
    """Absorbed MLA decode (_mla_decode) for every attention module with transformers' latent-cache expand_kv
    (glm4_moe_lite, glm_moe_dsa, glm5_next, deepseek_v3/v32, ...), a dense kv_b_proj (dequant_mla_kv) and sdpa
    attention: the class's expand_kv passes the latents through and the sdpa wrapper (install_decode_attention, needed)
    handles them. Returns the number of modules switched."""
    n = 0
    for m in model.modules():
        cls = type(m)
        if (not hasattr(cls, "expand_kv") or not isinstance(getattr(m, "kv_b_proj", None), torch.nn.Linear)
                or getattr(getattr(m, "config", None), "_attn_implementation", None) != "sdpa"
                or not all(hasattr(m, a) for a in ("qk_nope_head_dim", "v_head_dim"))):
            continue
        if cls.expand_kv is not _latent_expand_kv:
            cls._mla_expand = cls.expand_kv
            cls.expand_kv = _latent_expand_kv
        m._mla_absorb = True
        n += 1
    return n


def install_decode_attention() -> bool:
    """Wrap the registered "sdpa" attention (every model using it picks the wrapper up at call time)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cur = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(cur, "_moe_tiering", False):
        return False
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _decode_attention(cur)
    return True


# linear -> float softmax -> top-k -> renorm (norm_topk_prob, default on), checked equal
SOFTMAX_TOPK_ROUTERS = {"Qwen3MoeTopKRouter", "Qwen3_5MoeTopKRouter"}


def fuse_routers(model) -> int:
    """One-token routers through moe_tiering.router_kernels (two kernels instead of a vendor GEMV, a sort and glue), for
    the router classes in SOFTMAX_TOPK_ROUTERS; other tokens counts and classes keep their forward. Returns the
    number of routers patched now."""
    from .router_kernels import route_sigmoid_topk, route_softmax_topk

    fused = torch.compiler.disable(route_softmax_topk)
    fused_sig = torch.compiler.disable(route_sigmoid_topk)
    n = 0
    for r in arch.router_modules(model):
        if getattr(r, "_mt_fused", False):
            continue
        prev = r.forward  # the class forward, or arch.sort_router_topk's bound method
        if type(r).__name__ in SOFTMAX_TOPK_ROUTERS:
            def fwd(hidden_states, _r=r, _prev=prev):
                h = hidden_states.reshape(-1, _r.hidden_dim)
                if h.shape[0] == 1 and h.is_cuda:
                    return fused(h, _r.weight, _r.top_k, bool(getattr(_r, "norm_topk_prob", True)))
                return _prev(hidden_states)
        elif _sigmoid_noaux(r):
            def fwd(hidden_states, _r=r, _prev=prev):
                h = hidden_states.reshape(-1, _r.hidden_dim)
                if h.shape[0] == 1 and h.is_cuda:
                    return fused_sig(h, _r.weight, _r.e_score_correction_bias, _r.top_k, bool(_r.norm_topk_prob),
                                     _r.routed_scaling_factor)
                return _prev(hidden_states)
        else:
            continue
        r.forward = fwd
        r._mt_fused = True
        n += 1
    return n


# sigmoid -> + e_score_correction_bias -> top-k -> renorm -> x routed_scaling_factor (DeepSeek-V3's noaux_tc), fused
# when no group limit applies (one group, or every group kept); checked against the class forward (router_check.py)
SIGMOID_TOPK_ROUTERS = {"Glm4MoeLiteTopkRouter", "DeepseekV3TopkRouter", "GlmMoeDsaTopkRouter", "Glm4MoeTopkRouter"}


def _sigmoid_noaux(r) -> bool:
    if type(r).__name__ not in SIGMOID_TOPK_ROUTERS:
        return False
    groups = getattr(r, "num_group", getattr(r, "n_group", None))
    need = ("weight", "e_score_correction_bias", "top_k", "hidden_dim", "norm_topk_prob", "routed_scaling_factor")
    return all(hasattr(r, a) for a in need) and groups is not None and (groups == 1 or r.topk_group == groups)


def install_capture_safe_masks() -> bool:
    """transformers' eager_mask builds its 0.0 with torch.tensor(0.0, device=...), a host-to-device copy that a CUDA
    graph capture refuses (gpt-oss, Gemma-4: eager attention for sinks / softcapping). Same values from
    torch.zeros((), device=...) (a fill kernel), registered for "eager" attention masks."""
    import transformers.masking_utils as mu

    cur = mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    if getattr(cur, "_moe_tiering", False):
        return False
    sdpa_mask = mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def eager_mask(*args, dtype: torch.dtype = torch.float32, **kwargs):
        kwargs.pop("allow_is_causal_skip", None)
        mask = sdpa_mask(*args, allow_is_causal_skip=False, **kwargs)
        if mask is not None:
            mask = torch.where(mask, torch.zeros((), device=mask.device, dtype=dtype), torch.finfo(dtype).min)
        return mask

    eager_mask._moe_tiering = True
    mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask
    return True


def fuse_small_linears(model, max_out: int = 512) -> int:
    """One-token forward of the small floating-point nn.Linear modules in the decoder layers (gates, GatedDeltaNet's
    a / b projections: 1-32 outputs) as a float32 product and sum: inside a compiled layer one fused reduction instead
    of a vendor GEMM call (MI250: 24-67 us each for 1 x 2048 and 32 x 2048). F.linear's numerics: float32 sums of the
    exact products, rounded once to the activation dtype. Returns the number of modules patched now."""
    n = 0
    for layer in arch.decoder_layers(model):
        for m in layer.modules():
            if (type(m) is not torch.nn.Linear or m.out_features > max_out or getattr(m, "_mt_small", False)
                    or not m.weight.is_floating_point() or m.weight.dtype == torch.float32):
                continue
            orig = m.forward

            def fwd(x, _m=m, _o=orig):
                if x.dim() < 2 or x.shape[-2] != 1:
                    return _o(x)
                y = (x.float().unsqueeze(-2) * _m.weight.float()).sum(-1)
                if _m.bias is not None:
                    y = y + _m.bias.float()
                return y.to(x.dtype)

            m.forward = fwd
            m._mt_small = True
            n += 1
    return n


def dequant_mla_kv(model, names=None) -> int:
    """MLA models cache the compressed latent and run kv_b_proj over every cached position at each decode step (a
    T = cache-length GEMM per layer: the k4row GEMM took 423 us per layer on GLM-4.7-Flash, an MI250). The module
    becomes a bf16 nn.Linear holding its own dequantized weights (the same quantized values; ~6.5 MB per layer on
    GLM-4.7-Flash, counted by autoconfig.dense_facts), so that GEMM runs on the vendor BLAS. Returns the number of modules replaced."""
    from .autoconfig import BF16_DECODE_MODULES
    from .klinear import KLinear

    names = BF16_DECODE_MODULES if names is None else names
    n = 0
    for name, m in list(model.named_modules()):
        if not isinstance(m, KLinear) or name.rsplit(".", 1)[-1] not in names:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        lin = torch.nn.Linear(m.in_features, m.out_features, bias=m.bias is not None,
                              device=m.qweight.device, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(m.dequantize(torch.float32).to(torch.bfloat16))
            if m.bias is not None:
                lin.bias.copy_(m.bias.to(torch.bfloat16))
        lin.requires_grad_(False)
        setattr(parent, name.rsplit(".", 1)[-1], lin)
        n += 1
    return n


def _pooled_states_static(self, packed_states):
    """transformers 5.19 Glm5NextTextIndexer.get_pooled_states (modeling_glm5_next :936-1009) without its last line's
    boolean filter `keep = pool_valid.any(0); pool_keys[:, keep]`: a data-dependent shape a captured graph cannot
    hold. Every pool is returned in order; the unfilled static-cache pools are already invalid (pool_valid False),
    and forward masks invalid pools to finfo.min before its top-k and drops picked invalid pools (selected_valid), so
    the selected set is the same (select_k = min(index_topk // kpool, pools) only grows by pools that are masked)."""
    keys, gate_scores, valid_keys = torch.split(packed_states, [self.head_dim, self.head_dim, 1], dim=-1)
    valid_keys = valid_keys.bool().squeeze(-1)
    batch_size, seq_len = keys.shape[:2]
    number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
    device = keys.device
    first_key = torch.where(valid_keys.any(-1), valid_keys.long().argmax(-1),
                            torch.full((batch_size,), seq_len, dtype=torch.long, device=device))
    pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=device)
    pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
    pool_indices = first_key[:, None, None] + pool_offsets
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]
    safe_indices = pool_indices.clamp(0, seq_len - 1)
    grouped_keys = keys[batch_idx, safe_indices]
    grouped_gate_scores = gate_scores[batch_idx, safe_indices]
    grouped_valid_keys = valid_keys[batch_idx, safe_indices]
    grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
    pool_valid = grouped_valid_keys.all(-1)
    pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)
    logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
    logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
    probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
    pool_keys = (probabilities * grouped_keys).sum(dim=2)
    return pool_keys, pool_indices, pool_valid


def install_static_indexer() -> bool:
    """GLM-5.3-Flash (glm5_next) graph capture: the indexer's pooled states without the boolean filter
    (_pooled_states_static). Returns True when installed now."""
    try:
        from transformers.models.glm5_next import modeling_glm5_next as m
    except Exception:  # noqa: BLE001 - transformers without glm5_next
        return False
    cls = getattr(m, "Glm5NextTextIndexer", None)
    if cls is None or cls.get_pooled_states is _pooled_states_static:
        return False
    cls.get_pooled_states = _pooled_states_static
    return True
