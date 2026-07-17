#!/usr/bin/env python
"""v2 part-3: overlay per-slot attention onto decoded RGB frames.

Consumes what inference.py (KEY_ATTEND_MAP=...) writes under
<sample_dir> = .../key_attend_map/sample_XXXX/ :
  spatial/chunk{c}_block{b}_t{ts}.npy  -> {slot_mass [Lq, S], slots, F, H, W, ...}
  frames/chunk{c}.png                  -> one decoded RGB frame per chunk

Produces two families of figures under <sample_dir>/overlay/ :
  block{b}_t{ts}_{slot}.png  -- one per (block, timestep, slot); chunks concat
  AVG_{slot}.png             -- one per slot, averaged over ALL blocks & timesteps

In every figure the analyzed chunks are concatenated horizontally, each showing
that chunk's RGB frame with the slot's attention overlaid as a transparent color
intensity, normalized PER SLOT over all chunks in that figure.

The query axis (Lq = F*H*W, frame-major) is reshaped to (F, H, W, S) and the F
new-frames are averaged into a single (H, W) spatial map, then bilinearly
upsampled to the RGB frame size.

Usage:
    python analysis/plot_key_attend_map.py <sample_dir> [--out <dir>] [--cmap jet]
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_spatial(spatial_dir):
    """-> {(block, ts): {chunk: (hw_s [H,W,S], slots[list])}}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(spatial_dir, "*.npy"))):
        d = np.load(f, allow_pickle=True).item()
        sm = np.asarray(d["slot_mass"], dtype=np.float32)   # [Lq, S]
        F, H, W = int(d["F"]), int(d["H"]), int(d["W"])
        S = sm.shape[1]
        if sm.shape[0] != F * H * W:
            print(f"WARN: {os.path.basename(f)} Lq={sm.shape[0]} != F*H*W={F*H*W}, skipping")
            continue
        hw_s = sm.reshape(F, H, W, S).mean(axis=0)          # avg over new-frames -> [H, W, S]
        key = (int(d["block"]), int(d["timestep"]))
        out.setdefault(key, {})[int(d["chunk"])] = (hw_s, list(d["slots"]))
    return out


def load_frames(frames_dir):
    frames = {}
    for f in glob.glob(os.path.join(frames_dir, "chunk*.png")):
        c = int(os.path.basename(f)[len("chunk"):-len(".png")])
        frames[c] = plt.imread(f)   # [Hf, Wf, 3or4], float 0-1
    return frames


def all_slots(chunks_data):
    """Union of slot names across chunks, in first-seen order."""
    seen = []
    for c in sorted(chunks_data):
        for s in chunks_data[c][1]:
            if s not in seen:
                seen.append(s)
    return seen


def render_slot_overlay(chunks_data, slot, frames, out_png, title, cmap, alpha_gamma,
                        norm="max", pct=(5.0, 95.0)):
    """One figure for a single slot: each analyzed chunk's RGB frame with the
    slot's attention overlaid, concatenated horizontally, normalized over chunks.

    norm="max"        : vmin=0, vmax=global max (default).
    norm="percentile" : vmin,vmax = pct[0],pct[1] percentiles over all chunks'
                        values for this slot. Compresses the range so extreme
                        highs (e.g. sky) no longer wash out the low-attention
                        structure — makes weakly-attended regions readable.
    Returns True if a figure was written."""
    chunks = sorted(chunks_data)
    vals = [chunks_data[c][0][:, :, chunks_data[c][1].index(slot)]
            for c in chunks if slot in chunks_data[c][1]]
    if not vals:
        return False
    if norm == "percentile":
        allv = np.concatenate([v.ravel() for v in vals])
        vmin = float(np.percentile(allv, pct[0]))
        vmax = float(np.percentile(allv, pct[1]))
    else:
        vmin, vmax = 0.0, max(v.max() for v in vals)
    if vmax <= vmin:
        return False
    fig, axes = plt.subplots(1, len(chunks), squeeze=False,
                             figsize=(3.4 * len(chunks), 3.6))
    for i, c in enumerate(chunks):
        ax = axes[0][i]
        hw_s, names = chunks_data[c]
        frame = frames.get(c)
        extent = None
        if frame is not None:
            ax.imshow(frame)
            extent = [0, frame.shape[1], frame.shape[0], 0]
        if slot in names:
            att = hw_s[:, :, names.index(slot)]
            att_norm = np.clip((att - vmin) / (vmax - vmin), 0, 1)
            rgba = cmap(att_norm)
            rgba[..., 3] = att_norm ** alpha_gamma       # transparent where weak
            ax.imshow(rgba, extent=extent, interpolation="bilinear",
                      aspect="auto" if extent is not None else None)
            ax.set_title(f"chunk {c}", fontsize=9)
        else:
            ax.set_title(f"chunk {c} (no {slot})", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def average_over_block_time(data):
    """data {(block,ts): {chunk: (hw_s, slots)}} -> {chunk: (hw_s_mean, slots)}
    averaging each chunk's spatial map over all (block, timestep) it appears in."""
    items = defaultdict(list)
    for (_block, _ts), cd in data.items():
        for chunk, val in cd.items():
            items[chunk].append(val)
    avg = {}
    for chunk, vals in items.items():
        slots0 = vals[0][1]
        # Same chunk => same slot layout across blocks/timesteps; guard anyway.
        same = [hw for (hw, sl) in vals if sl == slots0]
        avg[chunk] = (np.stack(same, axis=0).mean(axis=0), slots0)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", help=".../key_attend_map/sample_XXXX")
    ap.add_argument("--out", default=None, help="output dir (default: <sample_dir>/overlay)")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--alpha-gamma", type=float, default=1.0,
                    help="raise normalized attention to this power for the alpha channel "
                         "(>1 makes weak attention more transparent)")
    ap.add_argument("--no-per-block", action="store_true",
                    help="skip the per-(block,timestep) overlays, keep only the averaged ones")
    ap.add_argument("--avg-percentile", action="store_true",
                    help="use percentile (robust) normalization for the AVG overlays; "
                         "default is max normalization (same as per-block)")
    ap.add_argument("--pct-low", type=float, default=5.0,
                    help="lower percentile for --avg-percentile normalization")
    ap.add_argument("--pct-high", type=float, default=95.0,
                    help="upper percentile for --avg-percentile normalization")
    args = ap.parse_args()

    spatial_dir = os.path.join(args.sample_dir, "spatial")
    frames_dir = os.path.join(args.sample_dir, "frames")
    if not os.path.isdir(spatial_dir):
        sys.exit(f"ERROR: {spatial_dir} not found")
    out_dir = args.out or os.path.join(args.sample_dir, "overlay")
    os.makedirs(out_dir, exist_ok=True)

    data = load_spatial(spatial_dir)
    frames = load_frames(frames_dir)
    if not data:
        sys.exit("ERROR: no spatial npy found")
    cmap = plt.get_cmap(args.cmap)
    n_made = 0

    # 1) per-(block, timestep) overlays.
    if not args.no_per_block:
        for (block, ts), chunks_data in sorted(data.items()):
            for slot in all_slots(chunks_data):
                out = os.path.join(out_dir, f"block{block:02d}_t{ts:04d}_{slot}.png")
                title = f"block {block}  t={ts}  slot={slot}   (overlay normalized over chunks)"
                if render_slot_overlay(chunks_data, slot, frames, out, title,
                                       cmap, args.alpha_gamma):
                    n_made += 1

    # 2) averaged over ALL blocks & timesteps: one figure per slot.
    chunk_avg = average_over_block_time(data)
    avg_norm = "percentile" if args.avg_percentile else "max"
    n_avg = 0
    for slot in all_slots(chunk_avg):
        out = os.path.join(out_dir, f"AVG_{slot}.png")
        if avg_norm == "percentile":
            title = (f"slot={slot}   (mean over all blocks & timesteps; "
                     f"percentile[{args.pct_low:g},{args.pct_high:g}] norm)")
        else:
            title = f"slot={slot}   (mean over all blocks & timesteps, normalized over chunks)"
        if render_slot_overlay(chunk_avg, slot, frames, out, title, cmap, args.alpha_gamma,
                               norm=avg_norm, pct=(args.pct_low, args.pct_high)):
            n_avg += 1

    print(f"overlay figures: {n_made} per-(block,timestep) + {n_avg} averaged -> {out_dir}")


if __name__ == "__main__":
    main()
