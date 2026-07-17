# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist


# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply_with_spatial_indices(
    x, grid_sizes, freqs, sink_tokens, num_rolled_tokens, num_recent_tokens, num_new_tokens,
    frame_seqlen, compressed_temporal_indices, compressed_spatial_indices, global_end_frame, use_block_rope,
    local_attn_size, sink_frames, recent_frames
):
    """
    Apply RoPE for Deep Forcing style cache: [Sink] + [Compressed] + [Recent] + [New]

    Block-Relativistic RoPE (same as main branch):
    - Cache: [0, 1, 2, ..., num_cache_frames - 1]
    - Query (New): [local_attn_size - new_frames, ..., local_attn_size - 1]

    Args:
        x: [B, L, H, D] - cache tokens (sink + compressed + recent + new)
        num_rolled_tokens: number of compressed tokens
        num_recent_tokens: number of recent tokens (always preserved)
        compressed_temporal_indices: [B, num_rolled_tokens] - original frame index for each compressed token
        compressed_spatial_indices: [B, num_rolled_tokens] - original spatial position (0 to frame_seqlen-1) for each compressed token

    VECTORIZED VERSION: Avoids Python loops for performance.
    """
    b, total_tokens, n, c2 = x.shape
    c = c2 // 2
    device = x.device
    f, h, w = int(grid_sizes[0, 0].item()), int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())

    # Split freqs
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freq_t, freq_h, freq_w = freqs_split[0], freqs_split[1], freqs_split[2]
    max_t_idx = freq_t.shape[0] - 1
    max_h_idx = freq_h.shape[0] - 1
    max_w_idx = freq_w.shape[0] - 1

    # Calculate frame counts
    new_frame_count = num_new_tokens // frame_seqlen if num_new_tokens > 0 else 0
    recent_frame_count = num_recent_tokens // frame_seqlen if num_recent_tokens > 0 else 0
    compressed_frame_count = (num_rolled_tokens + frame_seqlen - 1) // frame_seqlen if num_rolled_tokens > 0 else 0

    if use_block_rope:
        # Block-Relativistic RoPE: sequential indices starting from 0
        sink_temporal_base = 0
        compressed_temporal_base = sink_frames
        recent_temporal_base = sink_frames + compressed_frame_count
        new_temporal_base = local_attn_size - new_frame_count
    else:
        # Growing RoPE: use original global temporal indices
        sink_temporal_base = 0  # Sink frames always at global frames 0..sink_frames-1
        compressed_temporal_base = sink_frames  # Fallback when no compressed_temporal_indices
        recent_temporal_base = global_end_frame - new_frame_count - recent_frame_count
        new_temporal_base = global_end_frame - new_frame_count

    # Build temporal and spatial indices for all tokens at once
    temporal_indices_list = []
    h_indices_list = []
    w_indices_list = []

    # === SINK TOKENS ===
    if sink_tokens > 0:
        sink_token_indices = torch.arange(sink_tokens, device=device)
        sink_frame_indices = sink_token_indices // frame_seqlen
        sink_temporal = (sink_temporal_base + sink_frame_indices).clamp(0, max_t_idx)
        sink_local = sink_token_indices % frame_seqlen
        sink_h = (sink_local // w).clamp(0, max_h_idx)
        sink_w = (sink_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(sink_temporal)
        h_indices_list.append(sink_h)
        w_indices_list.append(sink_w)

    # === COMPRESSED TOKENS ===
    if num_rolled_tokens > 0:
        comp_token_indices = torch.arange(num_rolled_tokens, device=device)
        if not use_block_rope and compressed_temporal_indices is not None:
            # Growing RoPE: use actual global frame index per token
            comp_temporal = compressed_temporal_indices[0].long().clamp(0, max_t_idx)
        else:
            # Block-relativistic: sequential based on position in compressed region
            comp_temporal = (compressed_temporal_base + comp_token_indices // frame_seqlen).clamp(0, max_t_idx)

        # Spatial: use stored original positions if available
        if compressed_spatial_indices is not None:
            # Use first batch's spatial indices (assume same across batch)
            comp_local = compressed_spatial_indices[0].long()
        else:
            comp_local = comp_token_indices % frame_seqlen
        comp_h = (comp_local // w).clamp(0, max_h_idx)
        comp_w = (comp_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(comp_temporal)
        h_indices_list.append(comp_h)
        w_indices_list.append(comp_w)

    # === RECENT TOKENS ===
    if num_recent_tokens > 0:
        recent_token_indices = torch.arange(num_recent_tokens, device=device)
        recent_frame_in_region = recent_token_indices // frame_seqlen
        recent_temporal = (recent_temporal_base + recent_frame_in_region).clamp(0, max_t_idx)
        recent_local = recent_token_indices % frame_seqlen
        recent_h = (recent_local // w).clamp(0, max_h_idx)
        recent_w = (recent_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(recent_temporal)
        h_indices_list.append(recent_h)
        w_indices_list.append(recent_w)

    # === NEW TOKENS ===
    if num_new_tokens > 0:
        new_token_indices = torch.arange(num_new_tokens, device=device)
        new_frame_in_region = new_token_indices // frame_seqlen
        new_temporal = (new_temporal_base + new_frame_in_region).clamp(0, max_t_idx)
        new_local = new_token_indices % frame_seqlen
        new_h = (new_local // w).clamp(0, max_h_idx)
        new_w = (new_local % w).clamp(0, max_w_idx)
        temporal_indices_list.append(new_temporal)
        h_indices_list.append(new_h)
        w_indices_list.append(new_w)

    if not temporal_indices_list:
        return x

    # Concatenate all indices
    all_temporal = torch.cat(temporal_indices_list, dim=0)  # [L]
    all_h = torch.cat(h_indices_list, dim=0)  # [L]
    all_w = torch.cat(w_indices_list, dim=0)  # [L]
    L = all_temporal.shape[0]

    # Gather frequencies for all tokens at once
    freq_temporal = freq_t[all_temporal.long()]  # [L, c_t]
    freq_height = freq_h[all_h.long()]  # [L, c_h]
    freq_width = freq_w[all_w.long()]  # [L, c_w]

    # Concatenate frequencies: [L, c]
    all_freqs = torch.cat([freq_temporal, freq_height, freq_width], dim=-1)  # [L, c]
    all_freqs = all_freqs.unsqueeze(1)  # [L, 1, c]

    # Apply RoPE to x
    # x: [B, L, H, D] -> complex view
    x_view = x[:, :L].to(torch.float64).reshape(b, L, n, -1, 2)
    x_complex = torch.view_as_complex(x_view)  # [B, L, H, c]

    # Broadcast freqs across batch and heads
    result = x_complex * all_freqs.unsqueeze(0)  # [B, L, H, c]
    result = torch.view_as_real(result).flatten(3)  # [B, L, H, D]

    return result.type_as(x)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    """
    Apply causal RoPE (Rotary Position Embedding) to input tensor.
    
    Args:
        x: Input tensor of shape [B, L, num_heads, head_dim]
        grid_sizes: Tensor of shape [B, 3] containing (F, H, W)
        freqs: RoPE frequencies
        start_frame: Starting frame index for sequential RoPE (default: 0)
        relative_frame_indices: Optional tensor of shape [F] specifying explicit frame indices
                               for Block-Relativistic RoPE. If provided, overrides start_frame.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # Use relative_frame_indices if provided (Block-Relativistic RoPE),
        # otherwise use sequential indices starting from start_frame
        if relative_frame_indices is not None:
            # relative_frame_indices should be a tensor of shape [f] with explicit frame indices
            frame_indices = relative_frame_indices.long()
            freqs_temporal = freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
        else:
            freqs_temporal = freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1)
        
        freqs_i = torch.cat([
            freqs_temporal,
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def rope_apply_by_index(x, freqs, temporal_idx, h_idx, w_idx):
    """Apply RoPE to every token using explicit per-token (t, h, w) integer indices.

    Unlike ``causal_rope_apply`` (which derives spatial position from a token's
    physical (h, w) location inside a reshaped frame), this gathers the rotary
    frequency for each token independently. This lets content-merged memory
    tokens carry a *representative* spatial position that differs from their slot.

    Args:
        x:            [B, L, n, D] tokens to rotate (RoPE applies to Q/K only).
        freqs:        precomputed rotary table, [max_pos, D/2] complex.
        temporal_idx: [L] long, temporal (frame) index per token.
        h_idx:        [L] long, height index per token.
        w_idx:        [L] long, width index per token.

    Returns:
        [B, L, n, D] rotated tensor, same dtype as ``x``.
    """
    b, L, n, D = x.shape
    c = D // 2
    freq_t, freq_h, freq_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # Clamp to the precomputed table range (indices never exceed training range).
    ti = temporal_idx.long().clamp(0, freq_t.shape[0] - 1)
    hi = h_idx.long().clamp(0, freq_h.shape[0] - 1)
    wi = w_idx.long().clamp(0, freq_w.shape[0] - 1)
    freq = torch.cat([freq_t[ti], freq_h[hi], freq_w[wi]], dim=-1)  # [L, c] complex
    freq = freq.view(1, L, 1, c)
    x_c = torch.view_as_complex(x[:, :L].to(torch.float64).reshape(b, L, n, c, 2))
    out = torch.view_as_real(x_c * freq).flatten(3)
    return out.type_as(x)


# ---------------------------------------------------------------------------
# Cluster similarity instrumentation (analysis only).
# Disabled by default (log is None) so it has ZERO effect on normal runs.
# When enabled, `cluster_merge_update` records, for each evicted token, the
# cosine similarity to its top-1 (argmax) prototype, tagged by chunk index,
# transformer block index, and branch ('long'/'short').
# ---------------------------------------------------------------------------
_CLUSTER_SIM_LOG = None  # None disables; a list enables recording.


def enable_cluster_sim_logging(enabled=True):
    """Turn top-1 cosine-similarity logging on (fresh list) or off (None)."""
    global _CLUSTER_SIM_LOG
    _CLUSTER_SIM_LOG = [] if enabled else None


def get_cluster_sim_log():
    """Return the accumulated list of per-call similarity records (or None)."""
    return _CLUSTER_SIM_LOG


def _record_cluster_sim(chunk_idx, block_idx, branch, top1):
    """top1: [B, E] cosine sim of each evicted token to its assigned prototype."""
    if _CLUSTER_SIM_LOG is None:
        return
    t = top1.detach().float().reshape(-1)
    _CLUSTER_SIM_LOG.append({
        "chunk": int(chunk_idx),
        "block": int(block_idx),
        "branch": branch,
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "sum": float(t.sum()),
        "count": int(t.numel()),
    })


# ---------------------------------------------------------------------------
# Key-attribution instrumentation (analysis only).
# Disabled by default (log is None) so it has ZERO effect on normal runs.
# When enabled, the self-attention forward recomputes the query->key softmax
# (the fused attention() kernel does not expose weights) and records, for each
# (chunk, block), the average attention weight that each key GROUP receives:
#   sink / mem_long / mem_short / recent / curr(new)
# The groups partition the whole key window, so their weights sum to ~1.
# ---------------------------------------------------------------------------
_KEY_ATTEND_LOG = None  # None disables; a list enables recording.
# Cap queries used for the (analysis-only) softmax to keep it cheap; the group
# weights are averaged over queries so a uniform subsample is unbiased.
_KEY_ATTEND_MAX_QUERIES = 512
# Current denoising timestep, injected by the generation loop before each pass.
# None means "do not log this pass" (e.g. the clean-context cache-update rerun).
_KEY_ATTEND_TIMESTEP = None


def enable_key_attend_logging(enabled=True):
    """Turn per-group attention-weight logging on (fresh list) or off (None)."""
    global _KEY_ATTEND_LOG
    _KEY_ATTEND_LOG = [] if enabled else None


def get_key_attend_log():
    """Return the accumulated list of per-call attention-weight records (or None)."""
    return _KEY_ATTEND_LOG


def set_key_attend_timestep(t):
    """Tag subsequent attention calls with denoising timestep ``t`` (int), or
    pass None to suppress logging for the upcoming pass (clean-context rerun).
    Always safe to call; it is a no-op unless logging is enabled."""
    global _KEY_ATTEND_TIMESTEP
    _KEY_ATTEND_TIMESTEP = None if t is None else int(t)


def _record_key_attend(chunk_idx, block_idx, timestep, roped_q, roped_k, groups):
    """Recompute q->k softmax and log the mean weight mass per key group.

    Args:
        roped_q: [B, Lq, n, d] roped query used for this attention call.
        roped_k: [B, Lk, n, d] roped key actually attended (same tensor passed
                 to attention()); group indices below are positions within Lk.
        groups:  list of (name, start, end) half-open ranges over [0, Lk).
    """
    if _KEY_ATTEND_LOG is None:
        return
    B, Lq, n, d = roped_q.shape
    Lk = roped_k.shape[1]
    qh = roped_q.detach().permute(0, 2, 1, 3).float()   # [B, n, Lq, d]
    kh = roped_k.detach().permute(0, 2, 1, 3).float()   # [B, n, Lk, d]

    # Uniformly subsample queries (weights are averaged over them anyway).
    if Lq > _KEY_ATTEND_MAX_QUERIES:
        idx = torch.linspace(0, Lq - 1, _KEY_ATTEND_MAX_QUERIES, device=qh.device).long()
        qh = qh[:, :, idx, :]
        Lq_eff = _KEY_ATTEND_MAX_QUERIES
    else:
        Lq_eff = Lq

    scale = 1.0 / math.sqrt(d)
    # Average attention mass each key position receives, over batch/heads/queries.
    pos_mass = torch.zeros(Lk, device=qh.device, dtype=torch.float32)
    for bi in range(B):
        for hi in range(n):
            scores = torch.matmul(qh[bi, hi], kh[bi, hi].transpose(0, 1)) * scale  # [Lq_eff, Lk]
            probs = torch.softmax(scores, dim=-1)
            pos_mass += probs.sum(dim=0)
    pos_mass /= float(B * n * Lq_eff)

    # pos_mass[j] is the mean attention weight key position j receives (averaged
    # over batch/heads/queries). For each group we report BOTH:
    #   sum  = total attention mass to the group (the 5 groups sum to ~1)
    #   mean = per-token average = sum / (#tokens in group)  [size-normalized]
    # plus the min/max per-token weight within the group and the token count.
    rec = {"chunk": int(chunk_idx), "block": int(block_idx),
           "timestep": int(timestep), "kv_len": int(Lk)}
    for name, s, e in groups:
        s = max(0, min(int(s), Lk))
        e = max(0, min(int(e), Lk))
        if e > s:
            seg = pos_mass[s:e]
            rec[name] = {
                "sum": float(seg.sum()),
                "mean": float(seg.mean()),
                "min": float(seg.min()),
                "max": float(seg.max()),
                "count": int(e - s),
            }
        else:
            rec[name] = {"sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    _KEY_ATTEND_LOG.append(rec)


def cluster_merge_update(evicted_k, evicted_v, evicted_spatial,
                         proto_k, proto_v, proto_spatial, alpha,
                         sim_log_ctx=None):
    """Content-aware online clustering update of memory prototypes.

    Each evicted (un-roped) token is hard-assigned to the most cosine-similar
    memory prototype (over flattened heads); every prototype is then EMA-updated
    toward the mean of the tokens assigned to it. Prototypes that receive no
    token are left unchanged. This generalizes per-(h,w) EMA: it merges by
    *content* rather than by spatial slot. Deterministic (argmax + matmul
    aggregation; no scatter_add), so it is safe under
    ``torch.use_deterministic_algorithms(True)``.

    Args:
        evicted_k/evicted_v: [B, E, n, d] un-roped keys/values leaving the window.
        evicted_spatial:     [B, E] original spatial index (0..frame_seqlen-1).
        proto_k/proto_v:     [B, M, n, d] current memory prototypes.
        proto_spatial:       [B, M] float running-mean spatial position.
        alpha:               scalar EMA rate.

    Returns:
        new_proto_k, new_proto_v: [B, M, n, d]
        new_proto_spatial:        [B, M] float
    """
    B, E, n, d = evicted_k.shape
    M = proto_k.shape[1]

    # Similarity in float32 for stability/determinism (bf16 accumulation is lossy).
    e_feat = F.normalize(evicted_k.reshape(B, E, n * d).float(), dim=-1)
    p_feat = F.normalize(proto_k.reshape(B, M, n * d).float(), dim=-1)
    sim = torch.matmul(e_feat, p_feat.transpose(1, 2))        # [B, E, M]
    assign = sim.argmax(dim=-1)                               # [B, E]
    if sim_log_ctx is not None:
        # Record cosine sim of each evicted token to its top-1 prototype.
        _record_cluster_sim(*sim_log_ctx, sim.max(dim=-1).values)
    A = F.one_hot(assign, num_classes=M).float()             # [B, E, M]
    counts = A.sum(dim=1)                                     # [B, M]

    # Assigned-token means via matmul (deterministic), float32 accumulation.
    sum_k = torch.einsum('bem,bend->bmnd', A, evicted_k.float())
    sum_v = torch.einsum('bem,bend->bmnd', A, evicted_v.float())
    sum_s = torch.einsum('bem,be->bm', A, evicted_spatial.float())
    denom = counts.clamp(min=1.0)                             # avoid div-by-zero
    mean_k = sum_k / denom.view(B, M, 1, 1)
    mean_v = sum_v / denom.view(B, M, 1, 1)
    mean_s = sum_s / denom                                    # [B, M]

    # EMA only where a prototype actually received tokens.
    mask = (counts > 0).float()                              # [B, M]
    mk = mask.view(B, M, 1, 1)
    new_k = proto_k.float() + alpha * mk * (mean_k - proto_k.float())
    new_v = proto_v.float() + alpha * mk * (mean_v - proto_v.float())
    new_s = proto_spatial + alpha * mask * (mean_s - proto_spatial)
    return new_k.type_as(proto_k), new_v.type_as(proto_v), new_s


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 recent_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 use_block_rope=True,
                 compression_method='eviction',
                 ema_alpha_long=0.01,
                 ema_alpha_short=0.1,
                 ema_adaptive=False):
        """
        Args:
            sink_size: number of sink frames to preserve at the beginning
            recent_size: number of recent frames to preserve
            compression_method: 'eviction' (default) or 'ema'
                - eviction: simple FIFO eviction of oldest tokens
                - ema: [Sink] + [Long-term EMA] + [Short-term EMA] + [Recent] + [New]
                  Two EMA memories with different update rates
            ema_alpha_long: EMA update rate for long-term memory (small = slow update, default 0.01)
            ema_alpha_short: EMA update rate for short-term memory (large = fast update, default 0.1)
            ema_adaptive: if True, use per-token motion-based adaptive alpha (default False)
        """
        assert dim % num_heads == 0
        assert compression_method in ['eviction', 'ema', 'cluster']
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_block_rope = use_block_rope
        self.compression_method = compression_method
        self.ema_alpha_long = ema_alpha_long
        self.ema_alpha_short = ema_alpha_short
        self.ema_adaptive = ema_adaptive
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # Transformer block index, set by the parent model after block construction.
        # Used only for cluster-similarity instrumentation (per-block analysis).
        self.block_index = -1

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            num_new_frames = grid_sizes[0][0].item()  # F from grid_sizes
            
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            sink_frames = self.sink_size
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            kv_cache_frames = kv_cache_size // frame_seqlen
            
            # Determine if we're in Block-Relativistic RoPE mode (cache is full or rolling)
            is_rolling_mode = self.local_attn_size != -1 and (
                kv_cache["local_end_index"].item() + num_new_tokens > kv_cache_size
            )
            
            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # === ROLLING MODE ===
                # Calculate the number of tokens to evict/compress
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_evicted_frames = num_evicted_tokens // frame_seqlen

                # Create temporary k, v for computation - store UN-ROPED K
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()

                if self.compression_method in ('ema', 'cluster'):
                    # === EMA / CLUSTER MEMORY COMPRESSION ===
                    # 'ema':     memory slots are fixed (h,w) bins updated per-position.
                    # 'cluster': memory slots are content prototypes; evicted tokens are
                    #            merged into the most similar prototype (content-aware),
                    #            and each prototype tracks a representative spatial position.
                    is_cluster = self.compression_method == 'cluster'
                    # Cache structure: [Sink 3] + [Long-term EMA 1] + [Short-term EMA 1] + [Recent 4] + [New 3] = 12
                    # Long-term EMA: slow update (small alpha) - retains distant past
                    # Short-term EMA: fast update (large alpha) - captures recent trends
                    #
                    # If ema_adaptive=True: per-token alpha based on motion
                    #   motion[pos] = ||evicted[pos] - old_ema[pos]||
                    #   high motion → large alpha, low motion → small alpha

                    num_ema_frames = 2  # 1 long-term + 1 short-term
                    ema_tokens = num_ema_frames * frame_seqlen
                    current_local_end = kv_cache["local_end_index"].item()

                    # Recent fills the rest: total - sink - ema - new
                    dynamic_recent_frames = self.local_attn_size - self.sink_size - num_ema_frames - num_new_frames
                    recent_tokens = dynamic_recent_frames * frame_seqlen

                    # EMA positions (after sink)
                    ema_long_start = sink_tokens
                    ema_long_end = sink_tokens + frame_seqlen
                    ema_short_start = ema_long_end
                    ema_short_end = ema_short_start + frame_seqlen

                    # Recent starts after EMA
                    recent_start = ema_short_end
                    recent_end = current_local_end
                    actual_recent_tokens = recent_end - recent_start

                    # Always preserve sink frames
                    temp_k[:, :sink_tokens] = kv_cache["k"][:, :sink_tokens].clone()
                    temp_v[:, :sink_tokens] = kv_cache["v"][:, :sink_tokens].clone()

                    # Variables to pass to _apply_cache_updates
                    alpha_long_for_cache = self.ema_alpha_long  # scalar or tensor
                    alpha_short_for_cache = self.ema_alpha_short
                    evicted_k_for_cache = None
                    evicted_v_for_cache = None
                    # cluster-mode: final merged prototypes + representative spatial positions
                    cluster_long_k = cluster_long_v = None
                    cluster_short_k = cluster_short_v = None
                    proto_spatial_long_new = proto_spatial_short_new = None

                    if is_recompute:
                        # At recompute, cache already has updated layout from t=1000
                        # Just copy EMA and recent as-is, add new k/v
                        temp_k[:, ema_long_start:ema_short_end] = kv_cache["k"][:, ema_long_start:ema_short_end].clone()
                        temp_v[:, ema_long_start:ema_short_end] = kv_cache["v"][:, ema_long_start:ema_short_end].clone()

                        local_start_index = current_local_end - num_new_tokens
                        local_end_index = current_local_end
                        num_kept_recent = local_start_index - ema_short_end

                        if num_kept_recent > 0:
                            temp_k[:, ema_short_end:local_start_index] = kv_cache["k"][:, ema_short_end:local_start_index].clone()
                            temp_v[:, ema_short_end:local_start_index] = kv_cache["v"][:, ema_short_end:local_start_index].clone()

                        temp_k[:, local_start_index:local_end_index] = k
                        temp_v[:, local_start_index:local_end_index] = v
                    else:
                        # t=1000: Build new layout with EMA update
                        ema_initialized = "ema_initialized" in kv_cache and kv_cache["ema_initialized"]

                        if not ema_initialized and is_cluster:
                            # Cluster init: seed prototypes with the first recent frame
                            # (identity — prototype j starts at spatial position j).
                            seed_k = kv_cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                            seed_v = kv_cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                            temp_k[:, ema_long_start:ema_long_end] = seed_k
                            temp_v[:, ema_long_start:ema_long_end] = seed_v
                            temp_k[:, ema_short_start:ema_short_end] = seed_k
                            temp_v[:, ema_short_start:ema_short_end] = seed_v
                            cluster_long_k, cluster_long_v = seed_k, seed_v
                            cluster_short_k, cluster_short_v = seed_k, seed_v
                            init_spatial = torch.arange(frame_seqlen, device=k.device, dtype=torch.float32)
                            proto_spatial_long_new = init_spatial.unsqueeze(0).expand(b, -1).clone()
                            proto_spatial_short_new = proto_spatial_long_new.clone()
                        elif not ema_initialized:
                            # First time: initialize EMA
                            if num_evicted_tokens > 0:
                                evicted_k = kv_cache["k"][:, recent_start:recent_start + num_evicted_tokens]
                                evicted_v = kv_cache["v"][:, recent_start:recent_start + num_evicted_tokens]
                                if self.ema_adaptive and num_evicted_tokens >= frame_seqlen:
                                    # Per-position mean: preserves spatial structure
                                    num_evicted_frames = num_evicted_tokens // frame_seqlen
                                    evicted_k_mean = evicted_k[:, :num_evicted_frames * frame_seqlen].view(
                                        b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)
                                    evicted_v_mean = evicted_v[:, :num_evicted_frames * frame_seqlen].view(
                                        b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)
                                else:
                                    # Global mean (original behavior)
                                    evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                    evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                temp_k[:, ema_long_start:ema_long_end] = evicted_k_mean
                                temp_v[:, ema_long_start:ema_long_end] = evicted_v_mean
                                temp_k[:, ema_short_start:ema_short_end] = evicted_k_mean
                                temp_v[:, ema_short_start:ema_short_end] = evicted_v_mean
                                evicted_k_for_cache = evicted_k_mean
                                evicted_v_for_cache = evicted_v_mean
                            else:
                                temp_k[:, ema_long_start:ema_long_end] = kv_cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                                temp_v[:, ema_long_start:ema_long_end] = kv_cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                                temp_k[:, ema_short_start:ema_short_end] = kv_cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                                temp_v[:, ema_short_start:ema_short_end] = kv_cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                        else:
                            # Update EMA with evicted frames
                            old_ema_long_k = kv_cache["k"][:, ema_long_start:ema_long_end].clone()
                            old_ema_long_v = kv_cache["v"][:, ema_long_start:ema_long_end].clone()
                            old_ema_short_k = kv_cache["k"][:, ema_short_start:ema_short_end].clone()
                            old_ema_short_v = kv_cache["v"][:, ema_short_start:ema_short_end].clone()

                            if num_evicted_tokens > 0:
                                evicted_k = kv_cache["k"][:, recent_start:recent_start + num_evicted_tokens]
                                evicted_v = kv_cache["v"][:, recent_start:recent_start + num_evicted_tokens]

                                if is_cluster:
                                    # Content-aware merge into the most similar prototype.
                                    # Recent tokens are still un-merged, so their original
                                    # spatial index is simply position % frame_seqlen.
                                    evicted_spatial = (torch.arange(num_evicted_tokens, device=k.device) % frame_seqlen).unsqueeze(0).expand(b, -1)
                                    old_ps_long = kv_cache["proto_spatial_long"].clone()
                                    old_ps_short = kv_cache["proto_spatial_short"].clone()
                                    # Autoregressive chunk index (for similarity instrumentation).
                                    _sim_chunk_idx = int(current_start // frame_seqlen) // max(int(num_new_frames), 1)
                                    cluster_long_k, cluster_long_v, proto_spatial_long_new = cluster_merge_update(
                                        evicted_k, evicted_v, evicted_spatial,
                                        old_ema_long_k, old_ema_long_v, old_ps_long, self.ema_alpha_long,
                                        sim_log_ctx=(_sim_chunk_idx, self.block_index, 'long'))
                                    cluster_short_k, cluster_short_v, proto_spatial_short_new = cluster_merge_update(
                                        evicted_k, evicted_v, evicted_spatial,
                                        old_ema_short_k, old_ema_short_v, old_ps_short, self.ema_alpha_short,
                                        sim_log_ctx=(_sim_chunk_idx, self.block_index, 'short'))
                                    temp_k[:, ema_long_start:ema_long_end] = cluster_long_k
                                    temp_v[:, ema_long_start:ema_long_end] = cluster_long_v
                                    temp_k[:, ema_short_start:ema_short_end] = cluster_short_k
                                    temp_v[:, ema_short_start:ema_short_end] = cluster_short_v
                                else:
                                    if self.ema_adaptive and num_evicted_tokens >= frame_seqlen:
                                        # Per-position mean: preserves spatial structure
                                        num_evicted_frames = num_evicted_tokens // frame_seqlen
                                        evicted_k_mean = evicted_k[:, :num_evicted_frames * frame_seqlen].view(
                                            b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)
                                        evicted_v_mean = evicted_v[:, :num_evicted_frames * frame_seqlen].view(
                                            b, num_evicted_frames, frame_seqlen, n, d).mean(dim=1)

                                        # Long-term EMA: uniform alpha (stable global scene summary)
                                        alpha_long = self.ema_alpha_long

                                        # Short-term EMA: adaptive per-token alpha based on motion
                                        motion_short = (evicted_k_mean - old_ema_short_k).norm(dim=-1).mean(dim=-1)  # [B, frame_seqlen]
                                        motion_short_norm = motion_short / (motion_short.max(dim=-1, keepdim=True).values + 1e-8)  # [0, 1]

                                        alpha_short_min = self.ema_alpha_short * 0.1
                                        alpha_short_max = self.ema_alpha_short * 5.0
                                        alpha_short = (alpha_short_min + motion_short_norm * (alpha_short_max - alpha_short_min)).unsqueeze(-1).unsqueeze(-1)  # [B, frame_seqlen, 1, 1]
                                    else:
                                        # Original: global mean, scalar alpha
                                        evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                        evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                        alpha_long = self.ema_alpha_long
                                        alpha_short = self.ema_alpha_short

                                    # EMA update (works with both scalar and per-token alpha)
                                    new_ema_long_k = alpha_long * evicted_k_mean + (1 - alpha_long) * old_ema_long_k
                                    new_ema_long_v = alpha_long * evicted_v_mean + (1 - alpha_long) * old_ema_long_v
                                    new_ema_short_k = alpha_short * evicted_k_mean + (1 - alpha_short) * old_ema_short_k
                                    new_ema_short_v = alpha_short * evicted_v_mean + (1 - alpha_short) * old_ema_short_v

                                    temp_k[:, ema_long_start:ema_long_end] = new_ema_long_k
                                    temp_v[:, ema_long_start:ema_long_end] = new_ema_long_v
                                    temp_k[:, ema_short_start:ema_short_end] = new_ema_short_k
                                    temp_v[:, ema_short_start:ema_short_end] = new_ema_short_v

                                    # Store for _apply_cache_updates
                                    alpha_long_for_cache = alpha_long
                                    alpha_short_for_cache = alpha_short
                                    evicted_k_for_cache = evicted_k_mean
                                    evicted_v_for_cache = evicted_v_mean
                            else:
                                # No evicted tokens, keep old EMA
                                temp_k[:, ema_long_start:ema_long_end] = old_ema_long_k
                                temp_v[:, ema_long_start:ema_long_end] = old_ema_long_v
                                temp_k[:, ema_short_start:ema_short_end] = old_ema_short_k
                                temp_v[:, ema_short_start:ema_short_end] = old_ema_short_v

                        # FIFO shift Recent
                        k_recent = kv_cache["k"][:, recent_start + num_evicted_tokens:recent_end].clone()
                        v_recent = kv_cache["v"][:, recent_start + num_evicted_tokens:recent_end].clone()

                        num_kept_recent = k_recent.shape[1]
                        write_pos = ema_short_end
                        temp_k[:, write_pos:write_pos + num_kept_recent] = k_recent
                        temp_v[:, write_pos:write_pos + num_kept_recent] = v_recent
                        write_pos += num_kept_recent

                        local_end_index = write_pos + num_new_tokens
                        local_start_index = write_pos

                        temp_k[:, local_start_index:local_end_index] = k
                        temp_v[:, local_start_index:local_end_index] = v

                    # Apply RoPE (Block-Relativistic)
                    num_cache_frames = local_end_index // frame_seqlen
                    cache_grid_sizes = grid_sizes.clone()
                    cache_grid_sizes[0, 0] = num_cache_frames

                    query_relative_indices = torch.arange(
                        self.local_attn_size - num_new_frames,
                        self.local_attn_size,
                        device=q.device
                    )
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                    ).type_as(v)

                    cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                    if is_cluster:
                        # Spatial-aware RoPE: memory frames (long/short, right after the
                        # sink) carry each prototype's representative spatial position
                        # instead of their physical slot's (h, w). Sink/recent/new keep
                        # their native positions (identical to causal_rope_apply there).
                        w_grid = int(grid_sizes[0, 2].item())
                        cache_tok = temp_k[:, :local_end_index].view(
                            b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2)
                        tok = torch.arange(local_end_index, device=k.device)
                        fi = tok // frame_seqlen
                        spatial_local = (tok % frame_seqlen).clone()
                        # current valid proto spatial (freshly computed at t=1000, else cached)
                        cur_ps_long = proto_spatial_long_new if proto_spatial_long_new is not None else kv_cache["proto_spatial_long"]
                        cur_ps_short = proto_spatial_short_new if proto_spatial_short_new is not None else kv_cache["proto_spatial_short"]
                        ps_long = cur_ps_long[0].round().long().clamp(0, frame_seqlen - 1)
                        ps_short = cur_ps_short[0].round().long().clamp(0, frame_seqlen - 1)
                        if num_cache_frames > self.sink_size:
                            spatial_local[fi == self.sink_size] = ps_long
                        if num_cache_frames > self.sink_size + 1:
                            spatial_local[fi == self.sink_size + 1] = ps_short
                        temporal_idx = cache_relative_indices[fi]
                        h_idx = spatial_local // w_grid
                        w_idx = spatial_local % w_grid
                        roped_temp_k = rope_apply_by_index(
                            cache_tok, freqs, temporal_idx, h_idx, w_idx).type_as(v)
                    else:
                        roped_temp_k = causal_rope_apply(
                            temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                            cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                        ).type_as(v)

                    # Compute temporal/spatial indices for new tokens
                    new_token_positions = torch.arange(num_new_tokens, device=q.device)
                    new_temporal_indices = (current_start + new_token_positions) // frame_seqlen
                    new_spatial_indices = (current_start + new_token_positions) % frame_seqlen

                    cache_update_info = {
                        "action": "ema",
                        "sink_tokens": sink_tokens,
                        "ema_tokens": ema_tokens,
                        "local_start_index": local_start_index,
                        "local_end_index": local_end_index,
                        "write_start_index": local_start_index,
                        "write_end_index": local_end_index,
                        "new_k": k,
                        "new_v": v,
                        "new_q": q,
                        "new_temporal_indices": new_temporal_indices,
                        "new_spatial_indices": new_spatial_indices,
                        "current_end": current_end,
                        "is_recompute": is_recompute,
                        "num_evicted_tokens": num_evicted_tokens,
                        "alpha_long_tensor": alpha_long_for_cache,
                        "alpha_short_tensor": alpha_short_for_cache,
                        "evicted_k_mean": evicted_k_for_cache,
                        "evicted_v_mean": evicted_v_for_cache,
                        "ema_adaptive": self.ema_adaptive,
                        # cluster-mode: final merged prototypes + representative positions
                        "is_cluster": is_cluster,
                        "cluster_long_k": cluster_long_k,
                        "cluster_long_v": cluster_long_v,
                        "cluster_short_k": cluster_short_k,
                        "cluster_short_v": cluster_short_v,
                        "proto_spatial_long": proto_spatial_long_new,
                        "proto_spatial_short": proto_spatial_short_new,
                    }

                else:
                    # === SIMPLE EVICTION (FIFO) ===
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                    # Compute updated local indices
                    local_end_index = kv_cache["local_end_index"].item() + current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens

                    # Apply rolling update to the temporary cache
                    temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    # Insert new key/value into the temporary cache (UN-ROPED K!)
                    write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                    roped_offset = max(0, write_start_index - local_start_index)
                    write_len = max(0, local_end_index - write_start_index)
                    if write_len > 0:
                        temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                        temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                    # === RoPE Application for Eviction (Block-Relativistic, same as main) ===
                    num_cache_frames = local_end_index // frame_seqlen
                    cache_grid_sizes = grid_sizes.clone()
                    cache_grid_sizes[0, 0] = num_cache_frames

                    # Query: at end of window [local_attn_size - num_new_frames, ..., local_attn_size - 1]
                    query_relative_indices = torch.arange(
                        self.local_attn_size - num_new_frames,
                        self.local_attn_size,
                        device=q.device
                    )
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                    ).type_as(v)

                    # Cache: [0, 1, 2, ..., num_cache_frames - 1]
                    cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                    roped_temp_k = causal_rope_apply(
                        temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                        cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                    ).type_as(v)

                    # Compute temporal/spatial indices for new tokens
                    new_token_global_start = current_start + roped_offset
                    new_token_positions = torch.arange(write_len, device=q.device)
                    new_temporal_indices = (new_token_global_start + new_token_positions) // frame_seqlen
                    new_spatial_indices = (new_token_global_start + new_token_positions) % frame_seqlen

                    # Cache update info for eviction - store UN-ROPED K!
                    cache_update_info = {
                        "action": "roll_and_insert",
                        "sink_tokens": sink_tokens,
                        "num_rolled_tokens": num_rolled_tokens,
                        "num_evicted_tokens": num_evicted_tokens,
                        "local_start_index": local_start_index,
                        "local_end_index": local_end_index,
                        "write_start_index": write_start_index,
                        "write_end_index": local_end_index,
                        "new_k": k[:, roped_offset:roped_offset + write_len],
                        "new_v": v[:, roped_offset:roped_offset + write_len],
                        "new_q": q[:, roped_offset:roped_offset + write_len],  # For Deep Forcing
                        "new_temporal_indices": new_temporal_indices,  # [write_len]
                        "new_spatial_indices": new_spatial_indices,    # [write_len]
                        "current_end": current_end,
                        "is_recompute": is_recompute
                    }
            else:
                # === DIRECT INSERT MODE ===
                # Before cache is full, we can still use relative indices that grow sequentially
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation
                temp_k = kv_cache["k"].clone()  # UN-ROPED K
                temp_v = kv_cache["v"].clone()
                
                # Protect sink_tokens only during recomputation
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    # Store UN-ROPED K in cache
                    temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # === RoPE Application with Relative Indices (moviegen style) ===
                # Current frame position in the window
                current_frame_in_window = local_start_index // frame_seqlen

                # Query: apply RoPE with relative frame indices
                query_relative_indices = torch.arange(
                    current_frame_in_window,
                    current_frame_in_window + num_new_frames,
                    device=q.device
                )
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                ).type_as(v)

                # Cached K: apply RoPE dynamically
                num_cache_frames = local_end_index // frame_seqlen
                cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)

                cache_grid_sizes = grid_sizes.clone()
                cache_grid_sizes[0, 0] = num_cache_frames

                roped_temp_k = causal_rope_apply(
                    temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                    cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                ).type_as(v)

                # Compute temporal/spatial indices for new tokens
                new_token_global_start = current_start + roped_offset
                new_token_positions = torch.arange(write_len, device=q.device)
                new_temporal_indices = (new_token_global_start + new_token_positions) // frame_seqlen
                new_spatial_indices = (new_token_global_start + new_token_positions) % frame_seqlen

                # Save cache update info - store UN-ROPED K!
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len],  # UN-ROPED K!
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "new_q": q[:, roped_offset:roped_offset + write_len],  # For Deep Forcing
                    "new_temporal_indices": new_temporal_indices,  # [write_len]
                    "new_spatial_indices": new_spatial_indices,    # [write_len]
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }

            # Use roped K for attention computation
            # Limit V to same range as roped K (which has local_end_index tokens)
            temp_v_active = temp_v[:, :local_end_index]

            if sink_tokens > 0 and local_end_index > sink_tokens:
                # Concatenate sink tokens and local window tokens
                local_budget = self.max_attention_size - sink_tokens
                k_sink = roped_temp_k[:, :sink_tokens]
                v_sink = temp_v_active[:, :sink_tokens]
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = roped_temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v_active[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)

                    # Analysis-only: log per-group attention weight (no-op unless
                    # enabled). Logged on EVERY denoising pass tagged by the
                    # generation loop (_KEY_ATTEND_TIMESTEP is not None); the
                    # clean-context cache-update rerun sets it to None and is
                    # skipped. Only when the full window is attended so the group
                    # indices line up with k_cat.
                    if (_KEY_ATTEND_LOG is not None and _KEY_ATTEND_TIMESTEP is not None
                            and local_start_for_window == sink_tokens):
                        # Whether the cache currently holds mem prototypes. Do NOT
                        # infer this from cache_update_info["action"]: only the first
                        # denoising pass of a chunk takes the ROLLING path
                        # (action="ema"); the recompute passes (t=750/500/250) take
                        # the DIRECT-INSERT path (action="direct_insert") even though
                        # the cache still holds the same mem frames. Use action=="ema"
                        # (mem just (re)built this pass) OR ema_initialized (mem built
                        # in an earlier chunk and still resident).
                        _ka_action = cache_update_info.get("action") if cache_update_info else None
                        _ka_ema_init = ("ema_initialized" in kv_cache) and kv_cache["ema_initialized"]
                        _ka_has_mem = (
                            self.compression_method in ('ema', 'cluster')
                            and local_end_index >= sink_tokens + 2 * frame_seqlen
                            and ((_ka_action == "ema") or _ka_ema_init)
                        )
                        _ka_curr_start = local_end_index - num_new_tokens
                        _ka_groups = [("sink", 0, sink_tokens)]
                        if _ka_has_mem:
                            # Layout: [sink | mem_long 1f | mem_short 1f | recent | new]
                            _ka_ml_end = sink_tokens + frame_seqlen
                            _ka_ms_end = _ka_ml_end + frame_seqlen
                            _ka_groups += [
                                ("mem_long", sink_tokens, _ka_ml_end),
                                ("mem_short", _ka_ml_end, _ka_ms_end),
                                ("recent", _ka_ms_end, _ka_curr_start),
                            ]
                        else:
                            # Warm-up / no compressed memory yet: [sink | recent | new]
                            _ka_groups += [("recent", sink_tokens, _ka_curr_start)]
                        _ka_groups += [("curr", _ka_curr_start, local_end_index)]
                        _ka_chunk = int(current_start // frame_seqlen) // max(int(num_new_frames), 1)
                        _record_key_attend(_ka_chunk, self.block_index, _KEY_ATTEND_TIMESTEP,
                                           roped_query, k_cat, _ka_groups)
                else:
                    k_cat = k_sink
                    v_cat = v_sink

                x = attention(
                    roped_query,
                    k_cat,
                    v_cat,
                    deterministic=True
                )
            else:
                window_start = max(0, local_end_index - self.max_attention_size)

                x = attention(
                    roped_query,
                    roped_temp_k[:, window_start:local_end_index],
                    temp_v_active[:, window_start:local_end_index],
                    deterministic=True
                )

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 recent_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_block_rope=True,
                 compression_method='eviction',
                 ema_alpha_long=0.01,
                 ema_alpha_short=0.1,
                 ema_adaptive=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, recent_size, qk_norm, eps, use_block_rope, compression_method, ema_alpha_long, ema_alpha_short, ema_adaptive)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch)
        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True


    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 recent_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 use_block_rope=True,
                 compression_method='eviction',
                 ema_alpha_long=0.01,
                 ema_alpha_short=0.1,
                 ema_adaptive=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_block_rope = use_block_rope

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, recent_size, qk_norm, cross_attn_norm, eps, use_block_rope,
                                    compression_method, ema_alpha_long, ema_alpha_short, ema_adaptive)
            for _ in range(num_layers)
        ])
        # Tag each self-attention with its block index (for per-block instrumentation).
        for _bi, _blk in enumerate(self.blocks):
            _blk.self_attn.block_index = _bi
        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist


        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        
        For Block-Relativistic RoPE, this stores UN-ROPED K values in the cache.
        RoPE is applied dynamically during attention based on the token's current
        relative position in the sliding window.
        
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    new_q = update_info.get("new_q")  # For Deep Forcing
                    new_temporal_indices = update_info.get("new_temporal_indices")
                    new_spatial_indices = update_info.get("new_spatial_indices")

                    # Perform the rolling operation for K, V, Q
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    if "q" in cache:
                        cache["q"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["q"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    # Roll temporal/spatial indices (preserve original positions)
                    if "token_temporal_indices" in cache:
                        cache["token_temporal_indices"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["token_temporal_indices"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    if "token_spatial_indices" in cache:
                        cache["token_spatial_indices"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["token_spatial_indices"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    # Insert new key/value/query
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                        if new_q is not None and "q" in cache:
                            cache["q"][:, write_start_index:write_end_index] = new_q
                        # Store temporal/spatial indices for new tokens
                        if new_temporal_indices is not None and "token_temporal_indices" in cache:
                            cache["token_temporal_indices"][:, write_start_index:write_end_index] = new_temporal_indices
                        if new_spatial_indices is not None and "token_spatial_indices" in cache:
                            cache["token_spatial_indices"][:, write_start_index:write_end_index] = new_spatial_indices
                    

                elif update_info["action"] == "ema":
                    # EMA: [Sink] + [Long-term EMA] + [Short-term EMA] + [Recent] + [New]
                    sink_tokens = update_info["sink_tokens"]
                    ema_tokens = update_info["ema_tokens"]
                    write_start_index = update_info["write_start_index"]
                    write_end_index = update_info["write_end_index"]
                    local_start_index = update_info["local_start_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    new_q = update_info.get("new_q")
                    new_temporal_indices = update_info.get("new_temporal_indices")
                    new_spatial_indices = update_info.get("new_spatial_indices")
                    is_recompute = update_info.get("is_recompute", False)
                    num_evicted_tokens = update_info.get("num_evicted_tokens", 0)
                    # Alpha tensors from forward() (scalar or per-token tensor)
                    alpha_long = update_info.get("alpha_long_tensor", 0.01)
                    alpha_short = update_info.get("alpha_short_tensor", 0.1)
                    evicted_k_mean_from_fwd = update_info.get("evicted_k_mean")
                    evicted_v_mean_from_fwd = update_info.get("evicted_v_mean")
                    is_adaptive = update_info.get("ema_adaptive", False)

                    frame_seqlen = ema_tokens // 2  # 2 EMA frames
                    ema_long_start = sink_tokens
                    ema_long_end = sink_tokens + frame_seqlen
                    ema_short_start = ema_long_end
                    ema_short_end = ema_short_start + frame_seqlen
                    recent_start = ema_short_end

                    # Only do cache layout update at t=1000 (not recompute)
                    if not is_recompute:
                        current_local_end = cache["local_end_index"].item()

                        if update_info.get("is_cluster", False):
                            # Cluster: forward() already computed the merged prototypes
                            # (init or update); just write them and the proto positions.
                            clk = update_info.get("cluster_long_k")
                            if clk is not None:
                                cache["k"][:, ema_long_start:ema_long_end] = clk
                                cache["v"][:, ema_long_start:ema_long_end] = update_info["cluster_long_v"]
                                cache["k"][:, ema_short_start:ema_short_end] = update_info["cluster_short_k"]
                                cache["v"][:, ema_short_start:ema_short_end] = update_info["cluster_short_v"]
                            psl = update_info.get("proto_spatial_long")
                            if psl is not None:
                                pss = update_info["proto_spatial_short"]
                                cache["proto_spatial_long"] = psl
                                cache["proto_spatial_short"] = pss
                                # Mirror rounded positions into the integer index buffer.
                                if "token_spatial_indices" in cache:
                                    cache["token_spatial_indices"][:, ema_long_start:ema_long_end] = psl.round().long().clamp(0, frame_seqlen - 1)
                                    cache["token_spatial_indices"][:, ema_short_start:ema_short_end] = pss.round().long().clamp(0, frame_seqlen - 1)
                            cache["ema_initialized"] = True
                        else:
                            # Check if EMA is initialized
                            ema_initialized = "ema_initialized" in cache and cache["ema_initialized"]

                            if not ema_initialized:
                                # First time: use pre-computed values from forward()
                                if evicted_k_mean_from_fwd is not None:
                                    cache["k"][:, ema_long_start:ema_long_end] = evicted_k_mean_from_fwd
                                    cache["v"][:, ema_long_start:ema_long_end] = evicted_v_mean_from_fwd
                                    cache["k"][:, ema_short_start:ema_short_end] = evicted_k_mean_from_fwd
                                    cache["v"][:, ema_short_start:ema_short_end] = evicted_v_mean_from_fwd
                                elif num_evicted_tokens > 0:
                                    evicted_k = cache["k"][:, recent_start:recent_start + num_evicted_tokens]
                                    evicted_v = cache["v"][:, recent_start:recent_start + num_evicted_tokens]
                                    evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                    evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                    cache["k"][:, ema_long_start:ema_long_end] = evicted_k_mean
                                    cache["v"][:, ema_long_start:ema_long_end] = evicted_v_mean
                                    cache["k"][:, ema_short_start:ema_short_end] = evicted_k_mean
                                    cache["v"][:, ema_short_start:ema_short_end] = evicted_v_mean
                                else:
                                    cache["k"][:, ema_long_start:ema_long_end] = cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                                    cache["v"][:, ema_long_start:ema_long_end] = cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                                    cache["k"][:, ema_short_start:ema_short_end] = cache["k"][:, recent_start:recent_start + frame_seqlen].clone()
                                    cache["v"][:, ema_short_start:ema_short_end] = cache["v"][:, recent_start:recent_start + frame_seqlen].clone()
                                cache["ema_initialized"] = True
                            else:
                                # Update EMA using alpha tensors from forward()
                                if num_evicted_tokens > 0:
                                    old_ema_long_k = cache["k"][:, ema_long_start:ema_long_end].clone()
                                    old_ema_long_v = cache["v"][:, ema_long_start:ema_long_end].clone()
                                    old_ema_short_k = cache["k"][:, ema_short_start:ema_short_end].clone()
                                    old_ema_short_v = cache["v"][:, ema_short_start:ema_short_end].clone()

                                    # Use pre-computed evicted mean from forward() if available
                                    if evicted_k_mean_from_fwd is not None:
                                        evicted_k_mean = evicted_k_mean_from_fwd
                                        evicted_v_mean = evicted_v_mean_from_fwd
                                    else:
                                        evicted_k = cache["k"][:, recent_start:recent_start + num_evicted_tokens]
                                        evicted_v = cache["v"][:, recent_start:recent_start + num_evicted_tokens]
                                        evicted_k_mean = evicted_k.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)
                                        evicted_v_mean = evicted_v.mean(dim=1, keepdim=True).expand(-1, frame_seqlen, -1, -1)

                                    # alpha_long/alpha_short are either scalar or [B, frame_seqlen, 1, 1] tensor
                                    cache["k"][:, ema_long_start:ema_long_end] = alpha_long * evicted_k_mean + (1 - alpha_long) * old_ema_long_k
                                    cache["v"][:, ema_long_start:ema_long_end] = alpha_long * evicted_v_mean + (1 - alpha_long) * old_ema_long_v
                                    cache["k"][:, ema_short_start:ema_short_end] = alpha_short * evicted_k_mean + (1 - alpha_short) * old_ema_short_k
                                    cache["v"][:, ema_short_start:ema_short_end] = alpha_short * evicted_v_mean + (1 - alpha_short) * old_ema_short_v

                        # FIFO shift Recent
                        remaining_recent = local_start_index - ema_short_end
                        if remaining_recent > 0 and num_evicted_tokens > 0:
                            cache["k"][:, recent_start:recent_start + remaining_recent] = cache["k"][:, recent_start + num_evicted_tokens:current_local_end].clone()
                            cache["v"][:, recent_start:recent_start + remaining_recent] = cache["v"][:, recent_start + num_evicted_tokens:current_local_end].clone()
                            if "q" in cache:
                                cache["q"][:, recent_start:recent_start + remaining_recent] = cache["q"][:, recent_start + num_evicted_tokens:current_local_end].clone()

                    # Insert new tokens (always)
                    if write_end_index > write_start_index:
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                        if new_q is not None and "q" in cache:
                            cache["q"][:, write_start_index:write_end_index] = new_q
                        if new_temporal_indices is not None and "token_temporal_indices" in cache:
                            cache["token_temporal_indices"][:, write_start_index:write_end_index] = new_temporal_indices
                        if new_spatial_indices is not None and "token_spatial_indices" in cache:
                            cache["token_spatial_indices"][:, write_start_index:write_end_index] = new_spatial_indices


                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    new_q = update_info.get("new_q")  # For Deep Forcing
                    new_temporal_indices = update_info.get("new_temporal_indices")
                    new_spatial_indices = update_info.get("new_spatial_indices")

                    # Insert new key/value/query
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                        if new_q is not None and "q" in cache:
                            cache["q"][:, write_start_index:write_end_index] = new_q
                        # Store temporal/spatial indices for new tokens
                        if new_temporal_indices is not None and "token_temporal_indices" in cache:
                            cache["token_temporal_indices"][:, write_start_index:write_end_index] = new_temporal_indices
                        if new_spatial_indices is not None and "token_spatial_indices" in cache:
                            cache["token_spatial_indices"][:, write_start_index:write_end_index] = new_spatial_indices
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print("time embedding done")
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        # print("text embedding done")
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)


        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch,
        )
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            # print(f"block_index: {block_index}")
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                # print(f"forward checkpointing")
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                # print(f"forward no checkpointing")
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        kv_cache = kwargs.get('kv_cache', None)
        if kv_cache is not None:
            # Support attribute-based cache bypass for 2-GPU pipeline parallelism.
            # When dispatch_model's top-level hook copies kwargs across devices,
            # it creates new tensors for caches on other GPUs, breaking in-place
            # updates. Passing kv_cache=True signals to use model-attribute caches.
            if isinstance(kv_cache, bool) and kv_cache is True:
                kwargs['kv_cache'] = self._kv_cache_attr
                kwargs['crossattn_cache'] = self._crossattn_cache_attr
        return self._forward_inference(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
