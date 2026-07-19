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
import os
import numpy as np
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


# ---------------------------------------------------------------------------
# Key-attention MAP instrumentation (analysis v2; separate from v1 above).
# For a chosen set of chunk indices, this recomputes the FULL query->key
# attention matrix (head-averaged) on each denoising pass and, per (chunk,
# block, timestep):
#   (part 2) renders a query x key heatmap PNG immediately (the full matrix is
#            far too large to persist for every block/timestep), and
#   (part 3) saves the per-query, per-slot attention mass [Lq, num_slots] as a
#            small .npy so it can later be reshaped to (F, H, W, slots) and
#            overlaid on the decoded RGB frames.
# Disabled unless _KEY_ATTEND_MAP_DIR is set (via enable_key_attend_map).
# ---------------------------------------------------------------------------
_KEY_ATTEND_MAP_DIR = None       # output dir; None disables (denoising-pass v2).
_KEY_ATTEND_MAP_CHUNKS = set()   # only these autoregressive chunk indices.
# exp5 clean-pass variant: same attention-weight overlay, but recorded during the
# CLEAN-context pass (clean k/v prediction) instead of the denoising passes.
_KEY_ATTEND_MAP_CLEAN_DIR = None
_KEY_ATTEND_MAP_CLEAN_CHUNKS = set()


def enable_key_attend_map(chunks, out_dir):
    """Enable v2 attention-map logging for the given chunk indices, writing to
    <out_dir>/heatmap and <out_dir>/spatial. Pass out_dir=None to disable."""
    global _KEY_ATTEND_MAP_DIR, _KEY_ATTEND_MAP_CHUNKS
    _KEY_ATTEND_MAP_DIR = out_dir
    _KEY_ATTEND_MAP_CHUNKS = set(int(c) for c in chunks) if chunks else set()
    if out_dir is not None:
        os.makedirs(os.path.join(out_dir, "heatmap"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "spatial"), exist_ok=True)


def enable_key_attend_map_clean(chunks, out_dir):
    """exp5: enable the attention-weight overlay recorded on the CLEAN-context
    pass (clean k/v prediction), for the given chunk indices."""
    global _KEY_ATTEND_MAP_CLEAN_DIR, _KEY_ATTEND_MAP_CLEAN_CHUNKS
    _KEY_ATTEND_MAP_CLEAN_DIR = out_dir
    _KEY_ATTEND_MAP_CLEAN_CHUNKS = set(int(c) for c in chunks) if chunks else set()
    if out_dir is not None:
        os.makedirs(os.path.join(out_dir, "heatmap"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "spatial"), exist_ok=True)


def _render_slot_heatmap(pertoken, slot_names, F, frame_seqlen, chunk, block, timestep, out_dir):
    """part 2: query(rows) x slot(cols) heatmap of the per-token mean attention
    weight (slot mass / #tokens in slot). Small and fast to render, and the
    size-normalized values are comparable across slots (unlike the raw full
    query x key matrix, where every cell is ~1/kv_len and washes out)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Lq, S = pertoken.shape
    fig, ax = plt.subplots(figsize=(1.3 * S + 2.5, 8))
    im = ax.imshow(pertoken, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(S))
    ax.set_xticklabels(slot_names, rotation=45, ha="right")
    # new-frame boundaries along the query axis (frame-major F x H x W).
    for f in range(1, F):
        ax.axhline(f * frame_seqlen, color="w", lw=0.6)
    ax.set_ylabel("query token (frame-major: F x H x W)")
    ax.set_title(f"per-token mean attn — chunk {chunk}, block {block}, t={timestep}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="per-token mean attn weight")
    fig.tight_layout()
    out = os.path.join(out_dir, "heatmap",
                       f"chunk{chunk:04d}_block{block:02d}_t{timestep:04d}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _record_key_attend_map(chunk, block, timestep, roped_q, roped_k, groups,
                           num_new_frames, frame_seqlen, grid_sizes, out_dir=None):
    """Head-averaged attention aggregated PER SLOT (not the full q x k matrix):
    emit a part2 query x slot per-token-mean heatmap and a part3 per-slot spatial
    mass (sum). Only slot sums are accumulated, so no [Lq, Lk] matrix is kept.
    out_dir defaults to the denoising-pass dir; pass the clean dir for exp5."""
    if out_dir is None:
        out_dir = _KEY_ATTEND_MAP_DIR
    if out_dir is None:
        return
    B, Lq, n, d = roped_q.shape
    Lk = roped_k.shape[1]
    H = int(grid_sizes[0][1].item()); W = int(grid_sizes[0][2].item())
    S = len(groups)
    bounds = [(max(0, min(int(s), Lk)), max(0, min(int(e), Lk))) for (_, s, e) in groups]
    counts = torch.tensor([max(1, e - s) for (s, e) in bounds], dtype=torch.float32)
    qh = roped_q.detach().permute(0, 2, 1, 3).float()   # [B, n, Lq, d]
    kh = roped_k.detach().permute(0, 2, 1, 3).float()   # [B, n, Lk, d]
    scale = 1.0 / math.sqrt(d)
    # Accumulate only per-slot mass [Lq, S] (per head), never the full matrix.
    slot_sum_acc = torch.zeros(Lq, S, device=qh.device, dtype=torch.float32)
    for bi in range(B):
        for hi in range(n):
            scores = torch.matmul(qh[bi, hi], kh[bi, hi].transpose(0, 1)) * scale
            probs_h = torch.softmax(scores, dim=-1)     # [Lq, Lk]
            for si, (s, e) in enumerate(bounds):
                if e > s:
                    slot_sum_acc[:, si] += probs_h[:, s:e].sum(dim=1)
    slot_sum = (slot_sum_acc / float(B * n)).cpu()      # [Lq, S]  total mass per slot
    slot_names = [g[0] for g in groups]

    # part 2: per-token mean = slot mass / #tokens in slot -> query x slot heatmap.
    slot_pertoken = (slot_sum / counts).numpy()
    _render_slot_heatmap(slot_pertoken, slot_names, int(num_new_frames),
                         int(frame_seqlen), chunk, block, timestep, out_dir)

    # part 3: per-query slot mass (sum) for the RGB overlay.
    out = os.path.join(out_dir, "spatial",
                       f"chunk{chunk:04d}_block{block:02d}_t{timestep:04d}.npy")
    np.save(out, {"slot_mass": slot_sum.numpy().astype(np.float32), "slots": slot_names,
                  "F": int(num_new_frames), "H": H, "W": W,
                  "chunk": int(chunk), "block": int(block), "timestep": int(timestep)},
            allow_pickle=True)


# ---------------------------------------------------------------------------
# Memory-similarity MAP instrumentation (analysis exp4; separate from v2 above).
# During the CLEAN-context pass, for chosen chunk indices, record per new token
# the top-1 cosine similarity to the shadow long/short memory prototypes, for
# both key and value spaces -> 4 slots. Saved with the same [Lq, S] spatial
# schema as _record_key_attend_map so analysis/plot_key_attend_map.py is reused.
# Disabled unless _MEM_SIM_MAP_DIR is set. No effect on generation.
# ---------------------------------------------------------------------------
_MEM_SIM_MAP_DIR = None
_MEM_SIM_MAP_CHUNKS = set()
_MEM_SIM_SLOTS = ["key_long", "key_short", "value_long", "value_short"]


def enable_mem_sim_map(chunks, out_dir):
    """Enable exp4 memory-cos-sim logging for the given chunk indices."""
    global _MEM_SIM_MAP_DIR, _MEM_SIM_MAP_CHUNKS
    _MEM_SIM_MAP_DIR = out_dir
    _MEM_SIM_MAP_CHUNKS = set(int(c) for c in chunks) if chunks else set()
    if out_dir is not None:
        os.makedirs(os.path.join(out_dir, "spatial"), exist_ok=True)


def _top1_cossim(new_x, proto_x):
    """new_x: [B, Lq, n, d], proto_x: [B, M, n, d] -> [Lq] top-1 cos-sim (batch 0),
    over flattened heads (same space as cluster_merge_update)."""
    B, Lq = new_x.shape[0], new_x.shape[1]
    M = proto_x.shape[1]
    e = F.normalize(new_x.reshape(B, Lq, -1).float(), dim=-1)
    p = F.normalize(proto_x.reshape(B, M, -1).float(), dim=-1)
    sim = torch.matmul(e, p.transpose(1, 2))          # [B, Lq, M]
    return sim.max(dim=-1).values[0].detach().cpu()   # [Lq]


def _record_mem_sim_map(chunk, block, k_new, v_new,
                        smem_long_k, smem_long_v, smem_short_k, smem_short_v,
                        num_new_frames, frame_seqlen, grid_sizes):
    """Save per-new-token top-1 cos-sim to shadow memory: 4 slots
    {key,value}x{long,short}, [Lq, 4], reusing the key_attend_map .npy schema."""
    if _MEM_SIM_MAP_DIR is None:
        return
    H = int(grid_sizes[0][1].item()); W = int(grid_sizes[0][2].item())
    cols = [
        _top1_cossim(k_new, smem_long_k),
        _top1_cossim(k_new, smem_short_k),
        _top1_cossim(v_new, smem_long_v),
        _top1_cossim(v_new, smem_short_v),
    ]
    slot_mass = torch.stack(cols, dim=1).numpy().astype(np.float32)   # [Lq, 4]
    out = os.path.join(_MEM_SIM_MAP_DIR, "spatial",
                       f"chunk{chunk:04d}_block{block:02d}_t0000.npy")
    np.save(out, {"slot_mass": slot_mass, "slots": list(_MEM_SIM_SLOTS),
                  "F": int(num_new_frames), "H": H, "W": W,
                  "chunk": int(chunk), "block": int(block), "timestep": 0},
            allow_pickle=True)


def _parse_block_spec(spec):
    """Parse a block-index spec into a set of ints, or None meaning 'all blocks'.
    Accepts: None / "" / "all" -> None; "10,11,12" -> {10,11,12};
    "10-20" -> {10..20}; mixed "0,5,10-12" supported."""
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("", "all", "-1"):
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, bb = part.split("-", 1)
            out.update(range(int(a), int(bb) + 1))
        else:
            out.add(int(part))
    return out if out else None


def _refine_value_with_memory(k_new, v_new, proto_k, proto_v,
                              gate_mode="matched", tau=0.6, beta=0.1, alpha=1.0,
                              gate_fn="sigmoid", norm_restore=True,
                              aggregate="top1", temp=0.1):
    """exp5: refine new (clean recent) VALUE tokens toward memory (long prototypes).

    For each new token: match by KEY cos-sim to the top-1 prototype (j*), fetch
    that prototype's VALUE, and convex-blend it into the new value with a soft
    gate g that is LARGE where the new value poorly matches memory (low value
    cos-sim = likely artifact) and ~0 where it matches well.

      v_refined = (1 - g) * v_new + g * proto_v[j*]

    gate_mode (which value cos-sim vsim to gate on):
      "matched" : vsim = cos-sim(v_new, proto_v[j*])              (key-matched prototype's value)
      "top1"    : vsim = max_j cos-sim(v_new, proto_v[j])         (best value match, key-independent)

    gate_fn (how vsim maps to the blend weight g; low vsim -> larger g):
      "sigmoid" : g = alpha * sigmoid((tau - vsim) / beta)        (smooth; beta = temperature)
      "relu"    : g = alpha * clamp((tau - vsim) / tau, min=0)    (piecewise-linear; exactly 0 at vsim >= tau)
      "hard"    : g = alpha * (vsim < tau)                        ("replace" mode: step gate;
                  alpha=1.0 hard-replaces tokens with vsim<tau by pv_match, others untouched)

    aggregate (how the memory value target pv_target is formed):
      "top1" : pv_target = proto_v[argmax_j cos-sim(new_k, proto_k[j])]   (hard top-1, original)
      "attn" : pv_target = per-head cosine attention read from memory,
               softmax(cos(new_k_h, proto_k_h) / temp) @ proto_v_h        (soft aggregate)
    temp: temperature for the "attn" softmax (smaller = sharper -> closer to top-1;
      larger = flatter -> closer to the mean of prototype values, can blur).

    norm_restore: if True, after blending, rescale each head-vector back to the
      ORIGINAL v_new head-norm. The convex blend shrinks the norm (mixing two
      directions) which blurs the value; restoring the magnitude keeps the
      memory-guided direction while preserving the original energy. No-op where
      g==0 (unrefined tokens are returned exactly).

    k_new,v_new: [B, Lq, n, d] (un-roped); proto_k,proto_v: [B, M, n, d]. Key is
    NOT modified (only the value is refined). Returns [B, Lq, n, d] in v_new.dtype.
    """
    B, Lq, n, d = k_new.shape
    M = proto_k.shape[1]

    # --- form the memory value target pv_target [B, Lq, n, d] ---
    if aggregate == "attn":
        # per-head cosine attention read from memory, via the fused flash/SDPA
        # kernel (no [Lq, M] score matrix materialized). Pre-normalize q,k so the
        # dot-product is cosine; softmax_scale = 1/temp makes scores = cosine/temp.
        qh = F.normalize(k_new.float(), dim=-1).to(v_new.dtype)          # [B, Lq, n, d]
        kh = F.normalize(proto_k.float(), dim=-1).to(v_new.dtype)        # [B, M, n, d]
        pv_match = attention(qh, kh, proto_v, softmax_scale=1.0 / float(temp),
                             deterministic=True)                        # [B, Lq, n, d]
    else:  # "top1" (hard argmax match)
        kf = F.normalize(k_new.reshape(B, Lq, n * d).float(), dim=-1)
        pkf = F.normalize(proto_k.reshape(B, M, n * d).float(), dim=-1)
        jstar = torch.matmul(kf, pkf.transpose(1, 2)).argmax(dim=-1)    # [B, Lq] key top-1
        idx = jstar[:, :, None, None].expand(-1, -1, n, d)             # [B, Lq, n, d]
        pv_match = torch.gather(proto_v, 1, idx)                       # [B, Lq, n, d]

    vf = F.normalize(v_new.reshape(B, Lq, n * d).float(), dim=-1)
    if gate_mode == "top1":
        pvf = F.normalize(proto_v.reshape(B, M, n * d).float(), dim=-1)
        vsim = torch.matmul(vf, pvf.transpose(1, 2)).max(dim=-1).values  # [B, Lq]
    else:  # "matched" -> against the blend target pv_target
        pvm = F.normalize(pv_match.reshape(B, Lq, n * d).float(), dim=-1)
        vsim = (vf * pvm).sum(dim=-1)                                    # [B, Lq]

    if gate_fn == "hard":
        # step gate ("replace" mode): apply weight alpha exactly on tokens whose
        # value poorly matches memory (vsim < tau), 0 elsewhere. With alpha=1.0
        # this REPLACES those tokens' value by pv_match (delete + fill), leaving
        # well-matched tokens untouched; alpha<1.0 partially blends the selected.
        g = alpha * (vsim < tau).to(vsim.dtype)                          # [B, Lq]
    elif gate_fn == "relu":
        # piecewise-linear soft threshold; exactly 0 once vsim >= tau.
        g = alpha * torch.clamp((tau - vsim) / tau, min=0.0)             # [B, Lq]
    else:  # "sigmoid"
        g = alpha * torch.sigmoid((tau - vsim) / beta)                  # [B, Lq], low vsim -> high g
    g = g[:, :, None, None].to(v_new.dtype)
    v_hat = (1.0 - g) * v_new + g * pv_match.to(v_new.dtype)
    if norm_restore:
        # Restore each head-vector's original magnitude (blend keeps direction,
        # not energy). Computed in float32 for stability. No-op where g==0.
        num = v_new.float().norm(dim=-1, keepdim=True)                    # [B, Lq, n, 1]
        den = v_hat.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_hat = (v_hat.float() * (num / den)).to(v_new.dtype)
    return v_hat


def cluster_merge_update(evicted_k, evicted_v, evicted_spatial,
                         proto_k, proto_v, proto_spatial, alpha,
                         sim_log_ctx=None,
                         proto_count=None, proto_knorm=None,
                         want_count=False, want_knorm=False):
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
        proto_count:         [B, M] float per-prototype effective count n_i (exp3), or None.
        proto_knorm:         [B, M] float per-prototype running-mean raw key norm r_i (exp3), or None.
        want_count/want_knorm: only compute the corresponding new buffer when True;
                             otherwise the input is returned unchanged (no-op, default off).

    Returns:
        new_proto_k, new_proto_v: [B, M, n, d]
        new_proto_spatial:        [B, M] float
        new_proto_count:          [B, M] float (proto_count unchanged if want_count is False)
        new_proto_knorm:          [B, M] float (proto_knorm unchanged if want_knorm is False)
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

    # exp3 per-prototype scalars (computed only when requested; else passthrough).
    new_count = proto_count
    if want_count and proto_count is not None:
        # assigned prototype: n_i <- (1-alpha)*n_i + 1 ; unassigned: unchanged.
        new_count = proto_count + mask * (1.0 - alpha * proto_count)
    new_knorm = proto_knorm
    if want_knorm and proto_knorm is not None:
        # r_i <- (1-alpha)*r_i + alpha*mean(member raw key norm), assigned only.
        nrm = evicted_k.reshape(B, E, n * d).float().norm(dim=-1)   # [B, E] raw (pre-RoPE) key norm
        mean_nrm = torch.einsum('bem,be->bm', A, nrm) / denom       # [B, M]
        new_knorm = proto_knorm + alpha * mask * (mean_nrm - proto_knorm)

    return new_k.type_as(proto_k), new_v.type_as(proto_v), new_s, new_count, new_knorm


def _mem_logn_bias_vec(sink_tokens, frame_seqlen, Lk, n_long, n_short, dtype):
    """exp3 mem_logn_bias: additive attention-logit bias [B, Lk] equal to
    log(n_i) on the two memory frames (long/short) and 0 on sink/recent/current.
    n_long/n_short: [B, M] per-prototype effective counts."""
    B = n_long.shape[0]
    bias = torch.zeros(B, Lk, device=n_long.device, dtype=torch.float32)
    ml_end = sink_tokens + frame_seqlen
    ms_end = ml_end + frame_seqlen
    if ml_end <= Lk:
        bias[:, sink_tokens:ml_end] = torch.log(n_long.float().clamp_min(1.0))
    if ms_end <= Lk:
        bias[:, ml_end:ms_end] = torch.log(n_short.float().clamp_min(1.0))
    return bias.to(dtype)


def _sdpa_attn_with_bias(q, k, v, bias_bl):
    """SDPA attention with an additive per-key logit bias (FlashAttention exposes
    no bias hook, so mem_logn_bias forces this path). Same conventions as
    wan.modules.attention.attention: q/k/v are [B, L, n, d], scale = 1/sqrt(d)."""
    qs = q.transpose(1, 2)   # [B, n, Lq, d]
    ks = k.transpose(1, 2)
    vs = v.transpose(1, 2)
    attn_mask = bias_bl.view(bias_bl.shape[0], 1, 1, bias_bl.shape[1])  # broadcast heads/query
    out = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask)
    return out.transpose(1, 2).contiguous()   # [B, Lq, n, d]


def _renorm_mem_inplace(temp_k, sink_tokens, frame_seqlen, n_heads, head_dim,
                        r_long, r_short, eps=1e-6):
    """exp3 mem_key_renorm: rescale each memory prototype key to its running-mean
    raw norm r_i BEFORE RoPE. In-place on temp_k (a clone of the cache; the stored
    cache is not modified). r_long/r_short: [B, M] or None (skip that frame)."""
    B = temp_k.shape[0]
    for start, r in ((sink_tokens, r_long), (sink_tokens + frame_seqlen, r_short)):
        if r is None:
            continue
        end = start + frame_seqlen
        if end > temp_k.shape[1]:
            continue
        seg = temp_k[:, start:end]                                                 # [B, M, n, d]
        cn = seg.reshape(B, frame_seqlen, n_heads * head_dim).float().norm(dim=-1)  # [B, M]
        fac = (r.float() / (cn + eps)).to(temp_k.dtype).view(B, frame_seqlen, 1, 1)
        temp_k[:, start:end] = seg * fac


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
                 ema_adaptive=False,
                 mem_logn_bias=False,
                 mem_key_renorm=False,
                 mem_side_buffer=False,
                 mem_value_refine=False,
                 mem_value_refine_gate="matched",
                 mem_value_refine_tau=0.6,
                 mem_value_refine_beta=0.1,
                 mem_value_refine_alpha=1.0,
                 mem_value_refine_gate_fn="sigmoid",
                 mem_value_refine_blocks="all",
                 mem_value_refine_norm_restore=True,
                 mem_value_refine_aggregate="top1",
                 mem_value_refine_temp=0.1,
                 mem_value_refine_update_size=0,
                 clean_recent_attn_scale=1.0):
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
        # exp3 memory-attention corrections (both default off => identical to prior behavior)
        self.mem_logn_bias = mem_logn_bias
        self.mem_key_renorm = mem_key_renorm
        # exp4 shadow memory buffer (default off => identical to prior behavior).
        # When on: maintain long/short cluster prototypes in a side buffer (never
        # attended) for the clean-pass cos-sim overlay analysis.
        self.mem_side_buffer = mem_side_buffer
        # exp5 value refinement (default off => identical to prior behavior). When
        # on (requires eviction + mem_side_buffer): at the clean pass, refine the
        # new clean recent VALUE tokens toward their key-matched long prototype's
        # value via a soft-gated convex blend, before they are stored to the cache.
        self.mem_value_refine = mem_value_refine
        self.mem_value_refine_gate = mem_value_refine_gate
        self.mem_value_refine_tau = mem_value_refine_tau
        self.mem_value_refine_beta = mem_value_refine_beta
        self.mem_value_refine_alpha = mem_value_refine_alpha
        self.mem_value_refine_gate_fn = mem_value_refine_gate_fn
        # None => all blocks; else a set of block indices where refine is applied.
        self.mem_value_refine_blocks = _parse_block_spec(mem_value_refine_blocks)
        self.mem_value_refine_norm_restore = mem_value_refine_norm_restore
        self.mem_value_refine_aggregate = mem_value_refine_aggregate
        self.mem_value_refine_temp = mem_value_refine_temp
        # exp6: >0 splits the eviction past window into [update N frame (refined) |
        # recent 1 (raw)]; the frame(s) entering the update region each roll are
        # value-refined once and persisted to the cache. 0 = off (no-op).
        self.mem_value_refine_update_size = mem_value_refine_update_size
        # exp5: scale (<1 attenuates) attention to RECENT keys during the clean
        # k/v prediction pass. 1.0 = no change (default).
        self.clean_recent_attn_scale = clean_recent_attn_scale
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

            # exp4: on the CLEAN-context pass (_KEY_ATTEND_TIMESTEP is None) log the
            # top-1 cos-sim of this chunk's new clean K/V to the shadow long/short
            # prototypes. No-op unless mem_side_buffer + MEM_SIM_MAP are enabled and
            # the shadow memory has been seeded (so no effect on default runs).
            if (_MEM_SIM_MAP_DIR is not None and _KEY_ATTEND_TIMESTEP is None
                    and self.mem_side_buffer and kv_cache.get("smem_init", False)):
                _ms_chunk = int(current_start // frame_seqlen) // max(int(num_new_frames), 1)
                if _ms_chunk in _MEM_SIM_MAP_CHUNKS:
                    _record_mem_sim_map(
                        _ms_chunk, self.block_index, k, v,
                        kv_cache["smem_long_k"], kv_cache["smem_long_v"],
                        kv_cache["smem_short_k"], kv_cache["smem_short_v"],
                        num_new_frames, frame_seqlen, grid_sizes)

            # exp5: refine the new clean recent VALUE tokens toward the long memory
            # prototypes (key-matched, soft-gated convex blend). CACHE-ONLY: we do
            # NOT reassign `v`, so this block's attention (and every downstream
            # block's clean k/v prediction within this forward) is UNCHANGED. Only
            # the value written to the cache (cache_update_info["new_v"]) is
            # replaced by the refined value, per block independently, so the refined
            # value becomes recent for future chunks while the key stays original.
            # Clean pass only (_KEY_ATTEND_TIMESTEP is None). No-op unless the flag
            # is on and shadow memory is initialized.
            v_refined_for_cache = None
            if (self.mem_value_refine and _KEY_ATTEND_TIMESTEP is None
                    and self.compression_method == 'eviction'
                    and kv_cache.get("smem_init", False)
                    and (self.mem_value_refine_blocks is None
                         or self.block_index in self.mem_value_refine_blocks)):
                v_refined_for_cache = _refine_value_with_memory(
                    k, v, kv_cache["smem_long_k"], kv_cache["smem_long_v"],
                    gate_mode=self.mem_value_refine_gate,
                    tau=self.mem_value_refine_tau,
                    beta=self.mem_value_refine_beta,
                    alpha=self.mem_value_refine_alpha,
                    gate_fn=self.mem_value_refine_gate_fn,
                    norm_restore=self.mem_value_refine_norm_restore,
                    aggregate=self.mem_value_refine_aggregate,
                    temp=self.mem_value_refine_temp)

            # exp3: freshly-computed per-prototype count/knorm for this pass (set only
            # in the rolling cluster branch); None everywhere else. Declared here so
            # the shared attention block below can read them regardless of branch.
            mem_count_long_new = mem_count_short_new = None
            mem_knorm_long_new = mem_knorm_short_new = None

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
                    # exp3: per-prototype count n_i / key-norm r_i (None unless the
                    # corresponding flag is on; otherwise never staged/written/read).
                    mem_count_long_new = mem_count_short_new = None
                    mem_knorm_long_new = mem_knorm_short_new = None
                    _want_count = self.mem_logn_bias
                    _want_knorm = self.mem_key_renorm

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
                            # exp3 seed: each prototype starts as one token -> count=1,
                            # key-norm = the seed frame's raw per-prototype key norm.
                            if _want_count:
                                mem_count_long_new = torch.ones(b, frame_seqlen, device=k.device, dtype=torch.float32)
                                mem_count_short_new = mem_count_long_new.clone()
                            if _want_knorm:
                                seed_norm = seed_k.reshape(b, frame_seqlen, n * d).float().norm(dim=-1)  # [b, M]
                                mem_knorm_long_new = seed_norm.clone()
                                mem_knorm_short_new = seed_norm.clone()
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
                                    cluster_long_k, cluster_long_v, proto_spatial_long_new, mem_count_long_new, mem_knorm_long_new = cluster_merge_update(
                                        evicted_k, evicted_v, evicted_spatial,
                                        old_ema_long_k, old_ema_long_v, old_ps_long, self.ema_alpha_long,
                                        sim_log_ctx=(_sim_chunk_idx, self.block_index, 'long'),
                                        proto_count=(kv_cache["mem_count_long"] if _want_count else None),
                                        proto_knorm=(kv_cache["mem_knorm_long"] if _want_knorm else None),
                                        want_count=_want_count, want_knorm=_want_knorm)
                                    cluster_short_k, cluster_short_v, proto_spatial_short_new, mem_count_short_new, mem_knorm_short_new = cluster_merge_update(
                                        evicted_k, evicted_v, evicted_spatial,
                                        old_ema_short_k, old_ema_short_v, old_ps_short, self.ema_alpha_short,
                                        sim_log_ctx=(_sim_chunk_idx, self.block_index, 'short'),
                                        proto_count=(kv_cache["mem_count_short"] if _want_count else None),
                                        proto_knorm=(kv_cache["mem_knorm_short"] if _want_knorm else None),
                                        want_count=_want_count, want_knorm=_want_knorm)
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

                    # exp3 mem_key_renorm: rescale memory prototype keys to running-mean
                    # raw norm r_i BEFORE RoPE (cluster-mode only; no-op if flag off).
                    if self.mem_key_renorm and is_cluster and local_end_index >= sink_tokens + 2 * frame_seqlen:
                        _r_long = mem_knorm_long_new if mem_knorm_long_new is not None else kv_cache.get("mem_knorm_long")
                        _r_short = mem_knorm_short_new if mem_knorm_short_new is not None else kv_cache.get("mem_knorm_short")
                        _renorm_mem_inplace(temp_k, sink_tokens, frame_seqlen, n, d, _r_long, _r_short)

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
                        # exp3 per-prototype scalars (None unless flag on -> not written)
                        "mem_count_long": mem_count_long_new,
                        "mem_count_short": mem_count_short_new,
                        "mem_knorm_long": mem_knorm_long_new,
                        "mem_knorm_short": mem_knorm_short_new,
                    }

                else:
                    # === SIMPLE EVICTION (FIFO) ===
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                    # exp4 shadow memory: BEFORE the FIFO roll overwrites them, take the
                    # oldest recent tokens being evicted and fold them into the side
                    # long/short cluster prototypes (same cluster_merge_update algorithm;
                    # never attended). Only on the first pass (not is_recompute) so it
                    # updates once per chunk, mirroring cluster mode. No-op unless the
                    # flag is on -> identical to prior eviction behavior otherwise.
                    smem_long_k_new = smem_long_v_new = None
                    smem_short_k_new = smem_short_v_new = None
                    smem_sp_long_new = smem_sp_short_new = None
                    smem_init_new = None
                    if self.mem_side_buffer and not is_recompute and num_evicted_tokens > 0:
                        ev_k = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens]
                        # exp5(B): evict VALUE source = smem_src_v (ORIGINAL, pre-refine)
                        # so shadow memory tracks original content, not refined cache["v"].
                        # (Key never refined -> cache["k"] is already original.)
                        _ev_src_v = kv_cache["smem_src_v"] if "smem_src_v" in kv_cache else kv_cache["v"]
                        ev_v = _ev_src_v[:, sink_tokens:sink_tokens + num_evicted_tokens]
                        if not kv_cache.get("smem_init", False):
                            # Seed prototypes from the first recent frame (identity spatial),
                            # mirroring cluster init. Value seed also from the original source.
                            seed_k = kv_cache["k"][:, sink_tokens:sink_tokens + frame_seqlen].clone()
                            seed_v = _ev_src_v[:, sink_tokens:sink_tokens + frame_seqlen].clone()
                            smem_long_k_new, smem_long_v_new = seed_k, seed_v
                            smem_short_k_new, smem_short_v_new = seed_k.clone(), seed_v.clone()
                            _init_sp = torch.arange(frame_seqlen, device=k.device, dtype=torch.float32).unsqueeze(0).expand(b, -1).clone()
                            smem_sp_long_new = _init_sp
                            smem_sp_short_new = _init_sp.clone()
                            smem_init_new = True
                        else:
                            ev_spatial = (torch.arange(num_evicted_tokens, device=k.device) % frame_seqlen).unsqueeze(0).expand(b, -1)
                            smem_long_k_new, smem_long_v_new, smem_sp_long_new, _, _ = cluster_merge_update(
                                ev_k, ev_v, ev_spatial,
                                kv_cache["smem_long_k"], kv_cache["smem_long_v"], kv_cache["smem_spatial_long"],
                                self.ema_alpha_long)
                            smem_short_k_new, smem_short_v_new, smem_sp_short_new, _, _ = cluster_merge_update(
                                ev_k, ev_v, ev_spatial,
                                kv_cache["smem_short_k"], kv_cache["smem_short_v"], kv_cache["smem_spatial_short"],
                                self.ema_alpha_short)

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

                    # exp6: split the past window into [update N (refined) | recent 1 (raw)].
                    # After the roll, the frame(s) that just crossed recent->update occupy
                    # [u_end - num_new_tokens, u_end) (= the whole update region when
                    # roll==update_size). Refine ONLY those (recent/curr stay raw), and stage
                    # them so _apply_cache_updates persists them into cache["v"] (the smem_src_v
                    # mirror stays raw -> shadow memory folds the original). No-op unless the
                    # flag is on, shadow memory is ready, and the window is full.
                    refined_update_v = None
                    _u_start = sink_tokens
                    _u_end = sink_tokens + self.mem_value_refine_update_size * frame_seqlen
                    _r_start = _u_end - num_new_tokens
                    if (self.mem_value_refine_update_size > 0
                            and kv_cache.get("smem_init", False)
                            and (self.mem_value_refine_blocks is None
                                 or self.block_index in self.mem_value_refine_blocks)
                            and _r_start >= _u_start
                            and (local_end_index - num_new_tokens) >= _u_end + frame_seqlen):
                        refined_update_v = _refine_value_with_memory(
                            temp_k[:, _r_start:_u_end], temp_v[:, _r_start:_u_end],
                            kv_cache["smem_long_k"], kv_cache["smem_long_v"],
                            gate_mode=self.mem_value_refine_gate, tau=self.mem_value_refine_tau,
                            beta=self.mem_value_refine_beta, alpha=self.mem_value_refine_alpha,
                            gate_fn=self.mem_value_refine_gate_fn,
                            norm_restore=self.mem_value_refine_norm_restore,
                            aggregate=self.mem_value_refine_aggregate, temp=self.mem_value_refine_temp)
                        temp_v[:, _r_start:_u_end] = refined_update_v

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
                        # exp5(B): original value for the smem_src_v mirror (here == new_v,
                        # since curr is not refined on this rolling first-denoising pass).
                        "new_v_orig": v[:, roped_offset:roped_offset + write_len],
                        # exp6: refined update-region value to persist into cache["v"]
                        # after the roll (None unless flag on -> not written). smem_src_v
                        # is left raw so shadow memory still folds the original.
                        "refined_update_v": refined_update_v,
                        "update_write_start": _r_start,
                        "update_write_end": _u_end,
                        "new_q": q[:, roped_offset:roped_offset + write_len],  # For Deep Forcing
                        "new_temporal_indices": new_temporal_indices,  # [write_len]
                        "new_spatial_indices": new_spatial_indices,    # [write_len]
                        "current_end": current_end,
                        "is_recompute": is_recompute,
                        # exp4 shadow memory (None unless mem_side_buffer on -> not written)
                        "smem_long_k": smem_long_k_new, "smem_long_v": smem_long_v_new,
                        "smem_short_k": smem_short_k_new, "smem_short_v": smem_short_v_new,
                        "smem_spatial_long": smem_sp_long_new, "smem_spatial_short": smem_sp_short_new,
                        "smem_init": smem_init_new,
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

                # exp3 mem_key_renorm (recompute / direct-insert path): rescale memory
                # prototype keys to r_i before RoPE. mem frames are still resident here
                # (see recompute note); use the persisted buffer written at t=1000.
                if (self.mem_key_renorm and self.compression_method == 'cluster'
                        and kv_cache.get("ema_initialized")
                        and local_end_index >= sink_tokens + 2 * frame_seqlen):
                    _renorm_mem_inplace(temp_k, sink_tokens, frame_seqlen, n, d,
                                        kv_cache.get("mem_knorm_long"), kv_cache.get("mem_knorm_short"))

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

                # exp5 cache-only refine: the value WRITTEN to cache uses the refined
                # tensor (if produced this clean pass); attention above still used the
                # original `v` (via temp_v), so this block's output and every
                # downstream block's clean k/v are unchanged. Key stays original `k`.
                _v_for_cache = v_refined_for_cache if v_refined_for_cache is not None else v

                # Save cache update info - store UN-ROPED K!
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len],  # UN-ROPED K!
                    "new_v": _v_for_cache[:, roped_offset:roped_offset + write_len],
                    # exp5(B): ORIGINAL value for the smem_src_v mirror (pre-refine).
                    # Differs from new_v only on the clean pass (where new_v is refined).
                    "new_v_orig": v[:, roped_offset:roped_offset + write_len],
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
                    _ka_chunk = int(current_start // frame_seqlen) // max(int(num_new_frames), 1)
                    _ka_want_v1 = _KEY_ATTEND_LOG is not None and _KEY_ATTEND_TIMESTEP is not None
                    _ka_want_v2 = (_KEY_ATTEND_MAP_DIR is not None and _KEY_ATTEND_TIMESTEP is not None
                                   and _ka_chunk in _KEY_ATTEND_MAP_CHUNKS)
                    # exp5: same attention-weight overlay but on the CLEAN pass
                    # (clean k/v prediction), gated by _KEY_ATTEND_TIMESTEP is None.
                    _ka_want_clean = (_KEY_ATTEND_MAP_CLEAN_DIR is not None and _KEY_ATTEND_TIMESTEP is None
                                      and _ka_chunk in _KEY_ATTEND_MAP_CLEAN_CHUNKS)
                    if (_ka_want_v1 or _ka_want_v2 or _ka_want_clean) and local_start_for_window == sink_tokens:
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
                        if _ka_want_v1:
                            _record_key_attend(_ka_chunk, self.block_index, _KEY_ATTEND_TIMESTEP,
                                               roped_query, k_cat, _ka_groups)
                        if _ka_want_v2:
                            _record_key_attend_map(_ka_chunk, self.block_index, _KEY_ATTEND_TIMESTEP,
                                                   roped_query, k_cat, _ka_groups,
                                                   num_new_frames, frame_seqlen, grid_sizes)
                        if _ka_want_clean:
                            # clean pass has no denoising timestep; tag t=0.
                            _record_key_attend_map(_ka_chunk, self.block_index, 0,
                                                   roped_query, k_cat, _ka_groups,
                                                   num_new_frames, frame_seqlen, grid_sizes,
                                                   out_dir=_KEY_ATTEND_MAP_CLEAN_DIR)

                    # Build an optional additive attention-logit bias (routed through
                    # SDPA; FA2 exposes no bias hook). Two independent contributors,
                    # both only when the full window is attended (indices line up):
                    #  - exp3 mem_logn_bias: +log(n_i) on memory keys (cluster, mem resident)
                    #  - exp5 clean_recent_attn_scale: +log(scale) on RECENT keys during
                    #    the CLEAN pass only, to attenuate (scale<1) how much the clean
                    #    k/v prediction attends to recent tokens.
                    _attn_bias = None
                    _full_win = (local_start_for_window == sink_tokens)
                    _mem_resident = (self.compression_method in ('ema', 'cluster')
                                     and local_end_index >= sink_tokens + 2 * frame_seqlen
                                     and (kv_cache.get("ema_initialized")
                                          or (cache_update_info and cache_update_info.get("action") == "ema")))
                    if self.mem_logn_bias and self.compression_method == 'cluster' and _full_win and _mem_resident:
                        _n_long = mem_count_long_new if mem_count_long_new is not None else kv_cache.get("mem_count_long")
                        _n_short = mem_count_short_new if mem_count_short_new is not None else kv_cache.get("mem_count_short")
                        _attn_bias = _mem_logn_bias_vec(sink_tokens, frame_seqlen, k_cat.shape[1],
                                                        _n_long, _n_short, roped_query.dtype)
                    if (self.clean_recent_attn_scale != 1.0 and _KEY_ATTEND_TIMESTEP is None and _full_win):
                        _rec_start = (sink_tokens + 2 * frame_seqlen) if _mem_resident else sink_tokens
                        _rec_end = local_end_index - num_new_tokens
                        if _rec_end > _rec_start:
                            _rb = torch.zeros(k_cat.shape[1], device=roped_query.device, dtype=torch.float32)
                            _rb[_rec_start:_rec_end] = math.log(max(float(self.clean_recent_attn_scale), 1e-6))
                            _rb = _rb.unsqueeze(0).to(roped_query.dtype)   # [1, Lk] -> broadcast over batch
                            _attn_bias = _rb if _attn_bias is None else (_attn_bias + _rb)
                else:
                    k_cat = k_sink
                    v_cat = v_sink
                    _attn_bias = None

                if _attn_bias is not None:
                    x = _sdpa_attn_with_bias(roped_query, k_cat, v_cat, _attn_bias)
                else:
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
                 ema_adaptive=False,
                 mem_logn_bias=False,
                 mem_key_renorm=False,
                 mem_side_buffer=False,
                 mem_value_refine=False,
                 mem_value_refine_gate="matched",
                 mem_value_refine_tau=0.6,
                 mem_value_refine_beta=0.1,
                 mem_value_refine_alpha=1.0,
                 mem_value_refine_gate_fn="sigmoid",
                 mem_value_refine_blocks="all",
                 mem_value_refine_norm_restore=True,
                 mem_value_refine_aggregate="top1",
                 mem_value_refine_temp=0.1,
                 mem_value_refine_update_size=0,
                 clean_recent_attn_scale=1.0):
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
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, recent_size, qk_norm, eps, use_block_rope, compression_method, ema_alpha_long, ema_alpha_short, ema_adaptive, mem_logn_bias, mem_key_renorm, mem_side_buffer, mem_value_refine, mem_value_refine_gate, mem_value_refine_tau, mem_value_refine_beta, mem_value_refine_alpha, mem_value_refine_gate_fn, mem_value_refine_blocks, mem_value_refine_norm_restore, mem_value_refine_aggregate, mem_value_refine_temp, mem_value_refine_update_size, clean_recent_attn_scale)
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
                 ema_adaptive=False,
                 mem_logn_bias=False,
                 mem_key_renorm=False,
                 mem_side_buffer=False,
                 mem_value_refine=False,
                 mem_value_refine_gate="matched",
                 mem_value_refine_tau=0.6,
                 mem_value_refine_beta=0.1,
                 mem_value_refine_alpha=1.0,
                 mem_value_refine_gate_fn="sigmoid",
                 mem_value_refine_blocks="all",
                 mem_value_refine_norm_restore=True,
                 mem_value_refine_aggregate="top1",
                 mem_value_refine_temp=0.1,
                 mem_value_refine_update_size=0,
                 clean_recent_attn_scale=1.0):
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
                                    compression_method, ema_alpha_long, ema_alpha_short, ema_adaptive,
                                    mem_logn_bias, mem_key_renorm, mem_side_buffer,
                                    mem_value_refine, mem_value_refine_gate, mem_value_refine_tau,
                                    mem_value_refine_beta, mem_value_refine_alpha, mem_value_refine_gate_fn, mem_value_refine_blocks, mem_value_refine_norm_restore, mem_value_refine_aggregate, mem_value_refine_temp, mem_value_refine_update_size, clean_recent_attn_scale)
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
                    # exp5(B): roll the ORIGINAL-value mirror in lockstep.
                    if "smem_src_v" in cache:
                        cache["smem_src_v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["smem_src_v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
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
                        # exp5(B): mirror gets the ORIGINAL value.
                        if "smem_src_v" in cache:
                            cache["smem_src_v"][:, write_start_index:write_end_index] = \
                                update_info.get("new_v_orig", new_v)
                        if new_q is not None and "q" in cache:
                            cache["q"][:, write_start_index:write_end_index] = new_q
                        # Store temporal/spatial indices for new tokens
                        if new_temporal_indices is not None and "token_temporal_indices" in cache:
                            cache["token_temporal_indices"][:, write_start_index:write_end_index] = new_temporal_indices
                        if new_spatial_indices is not None and "token_spatial_indices" in cache:
                            cache["token_spatial_indices"][:, write_start_index:write_end_index] = new_spatial_indices

                    # exp6: persist the refined update-region value into cache["v"] AFTER
                    # the roll (which shifted raw values in) and the new_v insert. Only the
                    # update slot is overwritten; recent/curr/sink stay raw. smem_src_v is
                    # left raw (rolled above) so shadow memory still evicts the original.
                    _ruv = update_info.get("refined_update_v")
                    if _ruv is not None:
                        _rs = update_info["update_write_start"]
                        _re = update_info["update_write_end"]
                        cache["v"][:, _rs:_re] = _ruv

                    # exp4 shadow memory: persist updated side prototypes (only when
                    # mem_side_buffer produced them this pass; None on recompute/off).
                    if update_info.get("smem_long_k") is not None:
                        cache["smem_long_k"] = update_info["smem_long_k"]
                        cache["smem_long_v"] = update_info["smem_long_v"]
                        cache["smem_short_k"] = update_info["smem_short_k"]
                        cache["smem_short_v"] = update_info["smem_short_v"]
                        cache["smem_spatial_long"] = update_info["smem_spatial_long"]
                        cache["smem_spatial_short"] = update_info["smem_spatial_short"]
                        if update_info.get("smem_init"):
                            cache["smem_init"] = True

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
                            # exp3: persist per-prototype count / key-norm (only if flag on)
                            mcl = update_info.get("mem_count_long")
                            if mcl is not None:
                                cache["mem_count_long"] = mcl
                                cache["mem_count_short"] = update_info["mem_count_short"]
                            mkl = update_info.get("mem_knorm_long")
                            if mkl is not None:
                                cache["mem_knorm_long"] = mkl
                                cache["mem_knorm_short"] = update_info["mem_knorm_short"]
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
                        # exp5(B): mirror gets the ORIGINAL value (differs from new_v only
                        # on the clean pass, where new_v is the refined value).
                        if "smem_src_v" in cache:
                            cache["smem_src_v"][:, write_start_index:write_end_index] = \
                                update_info.get("new_v_orig", new_v)
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
