#!/usr/bin/env python
"""Plot per-group attention weight vs chunk index, per block AND per timestep.

The instrumentation in ``wan/modules/causal_model.py`` records, for every
denoising pass of each autoregressive chunk, how much attention weight each KEY
GROUP receives. The key window is partitioned into:

    sink | mem_long | mem_short | recent | curr(new)

Each record is tagged with (block, chunk, timestep) and, PER GROUP, stores:
    sum   = total attention mass to the group (the 5 groups sum to ~1)
    mean  = per-token average = sum / (#tokens)   [size-normalized]
    min   = smallest per-token weight in the group
    max   = largest  per-token weight in the group
    count = number of key tokens in the group

Because the groups differ in size (sink/curr = 3 frames, mem_* = 1 frame each),
`sum` favors big groups while `mean` is directly comparable across groups.

Outputs (per sample json), for each denoising timestep TS:
  out/t<TS>/blocks_grid_mean.png   per-token mean, one line per group, all blocks
  out/t<TS>/blocks_grid_sum.png    total mass (share, sums to 1), all blocks
  out/t<TS>/per_block/block_XX.png one subplot per group: mean line + min/max band
  out/t<TS>/key_attend_table.txt   per-block sum & mean table
  out/compare/block_XX.png         per group, lines = timesteps (mean)
  out/key_attend.csv               block,timestep,chunk,group,sum,mean,min,max,count

Usage:
    python analysis/plot_key_attend.py <key_attend_XXXX.json> [--out <dir>]
    python analysis/plot_key_attend.py outputs/.../key_attend/   # newest json in dir
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = ["sink", "mem_long", "mem_short", "recent", "curr"]
COLORS = {
    "sink": "tab:gray",
    "mem_long": "tab:red",
    "mem_short": "tab:orange",
    "recent": "tab:green",
    "curr": "tab:blue",
}
TS_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
METRICS = ["sum", "mean", "min", "max"]


def load_records(path):
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "key_attend_*.json")))
        if not cands:
            sys.exit(f"ERROR: no key_attend_*.json under {path}")
        path = cands[-1]
    with open(path) as f:
        return json.load(f), path


def _empty():
    return {"sum": [0.0, 0], "mean": [0.0, 0], "min": None, "max": None, "count": 0}


def aggregate(records):
    """-> {timestep: {block: {chunk: {group: {sum,mean,min,max,count}}}}}.

    Multiple records for the same (timestep, block, chunk) are combined:
    sum/mean averaged, min = min of mins, max = max of maxs.
    """
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(_empty))))
    for r in records:
        ts = int(r.get("timestep", -1))
        block, chunk = int(r["block"]), int(r["chunk"])
        for g in GROUPS:
            if g not in r:
                continue
            gv = r[g]
            if not isinstance(gv, dict):  # tolerate the old flat float format
                gv = {"sum": float(gv), "mean": float(gv), "min": float(gv),
                      "max": float(gv), "count": 0}
            a = acc[ts][block][chunk][g]
            a["sum"][0] += gv["sum"]; a["sum"][1] += 1
            a["mean"][0] += gv["mean"]; a["mean"][1] += 1
            a["min"] = gv["min"] if a["min"] is None else min(a["min"], gv["min"])
            a["max"] = gv["max"] if a["max"] is None else max(a["max"], gv["max"])
            a["count"] = gv["count"]
    out = {}
    for ts, blocks in acc.items():
        out[ts] = {}
        for block, chunks in blocks.items():
            out[ts][block] = {}
            for chunk, gs in chunks.items():
                out[ts][block][chunk] = {
                    g: {
                        "sum": v["sum"][0] / v["sum"][1] if v["sum"][1] else float("nan"),
                        "mean": v["mean"][0] / v["mean"][1] if v["mean"][1] else float("nan"),
                        "min": v["min"] if v["min"] is not None else float("nan"),
                        "max": v["max"] if v["max"] is not None else float("nan"),
                        "count": v["count"],
                    } for g, v in gs.items()
                }
    return out


def write_csv(agg, out_csv):
    with open(out_csv, "w") as f:
        f.write("block,timestep,chunk,group,sum,mean,min,max,count\n")
        for ts in sorted(agg):
            for block in sorted(agg[ts]):
                for chunk in sorted(agg[ts][block]):
                    for g in GROUPS:
                        v = agg[ts][block][chunk].get(g)
                        if v is None:
                            continue
                        f.write(f"{block},{ts},{chunk},{g},"
                                f"{v['sum']:.6f},{v['mean']:.6f},{v['min']:.6f},"
                                f"{v['max']:.6f},{v['count']}\n")


def block_means(agg_ts, metric):
    """agg_ts = {block: {chunk: {group: {metrics}}}} -> {block: {group: mean_over_chunks}}."""
    means = {}
    overall = {g: [0.0, 0] for g in GROUPS}
    for block in sorted(agg_ts):
        bm = {}
        for g in GROUPS:
            vals = [agg_ts[block][c][g][metric] for c in agg_ts[block]
                    if g in agg_ts[block][c] and agg_ts[block][c][g][metric] == agg_ts[block][c][g][metric]]
            if vals:
                bm[g] = sum(vals) / len(vals)
                overall[g][0] += sum(vals); overall[g][1] += len(vals)
        means[block] = bm
    means["ALL"] = {g: (overall[g][0] / overall[g][1] if overall[g][1] else float("nan"))
                    for g in GROUPS}
    return means


def save_table(agg_ts, out_txt, ts):
    sums = block_means(agg_ts, "sum")
    meansv = block_means(agg_ts, "mean")
    lines = [f"Per-block attention weight — timestep {ts}  (averaged over chunks)"]
    for label, table, fmt in [
            ("TOTAL MASS (share, sums to 1 across groups)", sums, "11.4f"),
            ("PER-TOKEN MEAN (size-normalized; ~1/kv_len scale)", meansv, "11.3e")]:
        header = f"{'block':>6}  " + "".join(f"{g:>11}" for g in GROUPS)
        lines += ["", label, header, "-" * len(header)]
        for block in [b for b in table if b != "ALL"] + ["ALL"]:
            bm = table[block]
            cells = "".join(
                (f"{bm[g]:>{fmt}}" if g in bm and bm[g] == bm[g] else f"{'-':>11}")
                for g in GROUPS)
            if block == "ALL":
                lines.append("-" * len(header))
            lines.append(f"{str(block):>6}  {cells}")
    text = "\n".join(lines)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    return sums, meansv


def plot_group_lines(ax, series, metric, title):
    """One line per group of series[chunk][group][metric]."""
    chunks = sorted(series)
    for g in GROUPS:
        ys = [series[c].get(g, {}).get(metric, float("nan")) for c in chunks]
        if all(y != y for y in ys):
            continue
        ax.plot(chunks, ys, color=COLORS[g], lw=1.5, marker=".", ms=3, label=g)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)


def render_overview(agg_ts, out_png, metric, tag):
    blocks = sorted(agg_ts)
    ncol = 5
    nrow = (len(blocks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for i, block in enumerate(blocks):
        ax = axes[i // ncol][i % ncol]
        plot_group_lines(ax, agg_ts[block], metric, f"block {block}")
        if i % ncol == 0:
            ax.set_ylabel(f"{metric} attn weight")
        if i // ncol == nrow - 1:
            ax.set_xlabel("chunk idx")
    for j in range(len(blocks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=9)
    fig.suptitle(f"Attention weight per key group ({metric}) vs chunk — {tag}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def render_perblock_bands(agg_ts, out_dir, tag):
    """Per block: one subplot per group, per-token mean line + min/max band."""
    os.makedirs(out_dir, exist_ok=True)
    for block in sorted(agg_ts):
        series = agg_ts[block]
        chunks = sorted(series)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        for gi, g in enumerate(GROUPS):
            ax = axes[gi // 3][gi % 3]
            me = [series[c].get(g, {}).get("mean", float("nan")) for c in chunks]
            mn = [series[c].get(g, {}).get("min", float("nan")) for c in chunks]
            mx = [series[c].get(g, {}).get("max", float("nan")) for c in chunks]
            if not all(y != y for y in me):
                ax.fill_between(chunks, mn, mx, color=COLORS[g], alpha=0.2, label="min–max")
                ax.plot(chunks, me, color=COLORS[g], lw=1.6, marker=".", ms=3, label="mean")
            ax.set_title(g, fontsize=10)
            ax.set_xlabel("chunk idx")
            ax.set_ylabel("per-token attn weight")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
        axes[1][2].axis("off")
        fig.suptitle(f"block {block} — per-token attention (mean + min/max) — {tag}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(out_dir, f"block_{block:02d}.png"), dpi=130)
        plt.close(fig)


def render_compare(agg, out_dir, timesteps, metric="mean"):
    """Per block: one subplot per group, lines = timesteps."""
    os.makedirs(out_dir, exist_ok=True)
    blocks = sorted({b for ts in agg for b in agg[ts]})
    tcolor = {ts: TS_COLORS[i % len(TS_COLORS)] for i, ts in enumerate(timesteps)}
    for block in blocks:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        for gi, g in enumerate(GROUPS):
            ax = axes[gi // 3][gi % 3]
            for ts in timesteps:
                series = agg.get(ts, {}).get(block, {})
                chunks = sorted(series)
                ys = [series[c].get(g, {}).get(metric, float("nan")) for c in chunks]
                if not chunks or all(y != y for y in ys):
                    continue
                ax.plot(chunks, ys, color=tcolor[ts], lw=1.4, marker=".", ms=3, label=f"t={ts}")
            ax.set_title(g, fontsize=10)
            ax.set_xlabel("chunk idx")
            ax.set_ylabel(f"{metric} attn weight")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            if gi == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, title="timestep")
        axes[1][2].axis("off")
        fig.suptitle(f"block {block}: per-group {metric} across denoising timesteps", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(out_dir, f"block_{block:02d}.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="key_attend_*.json file or a directory containing them")
    ap.add_argument("--out", default=None, help="output dir (default: <input_dir>/plots)")
    ap.add_argument("--no-compare", action="store_true", help="skip timestep-compare figures")
    args = ap.parse_args()

    records, src = load_records(args.input)
    print(f"loaded {len(records)} records from {src}")
    agg = aggregate(records)
    if not agg:
        sys.exit("ERROR: no usable records")

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(src)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(agg, os.path.join(out_dir, "key_attend.csv"))

    timesteps = sorted(agg, reverse=True)
    print(f"timesteps found: {timesteps}")
    for ts in timesteps:
        tsdir = os.path.join(out_dir, f"t{ts}")
        os.makedirs(tsdir, exist_ok=True)
        render_overview(agg[ts], os.path.join(tsdir, "blocks_grid_mean.png"),
                        metric="mean", tag=f"timestep {ts}")
        render_overview(agg[ts], os.path.join(tsdir, "blocks_grid_sum.png"),
                        metric="sum", tag=f"timestep {ts}")
        render_perblock_bands(agg[ts], os.path.join(tsdir, "per_block"), tag=f"timestep {ts}")
        save_table(agg[ts], os.path.join(tsdir, "key_attend_table.txt"), ts)
        print(f"[timestep {ts}] -> {tsdir}")

    if not args.no_compare:
        cmp_dir = os.path.join(out_dir, "compare")
        render_compare(agg, cmp_dir, timesteps, metric="mean")
        print(f"compare figures -> {cmp_dir}")


if __name__ == "__main__":
    main()
