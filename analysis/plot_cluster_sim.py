#!/usr/bin/env python
"""Plot top-1 cluster cosine similarity vs autoregressive chunk index, per block.

The instrumentation in ``wan/modules/causal_model.py`` records, for every evicted
token at each compression step, the cosine similarity to its top-1 (argmax)
prototype. Each record is one call = (chunk, block, branch) with the min / mean /
max over that step's evicted tokens (plus sum & count for exact re-aggregation).

This script aggregates those records per (block, branch, chunk) and draws, for
EACH transformer block, a separate graph of similarity vs chunk index:
  - x axis: autoregressive chunk index (deeper = later in the video)
  - y axis: top-1 cosine similarity   (mean line + min/max shaded band)

Usage:
    # after running inference with CLUSTER_SIM_LOG=1
    python analysis/plot_cluster_sim.py <cluster_sim_XXXX.json> [--branch long|short|both] [--out <dir>]
    python analysis/plot_cluster_sim.py outputs/.../cluster_sim/   # newest json in dir
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


def load_records(path):
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "cluster_sim_*.json")))
        if not cands:
            sys.exit(f"ERROR: no cluster_sim_*.json under {path}")
        path = cands[-1]
    with open(path) as f:
        return json.load(f), path


def aggregate(records, branch):
    """-> {block: {chunk: (min, mean, max)}} for the given branch.

    Robust to multiple records per (block, chunk) — e.g. several denoising
    steps hitting compression in one chunk — via weighted mean over counts.
    """
    acc = defaultdict(lambda: defaultdict(lambda: {"min": None, "max": None, "sum": 0.0, "cnt": 0}))
    for r in records:
        if branch != "both" and r["branch"] != branch:
            continue
        a = acc[r["block"]][r["chunk"]]
        a["min"] = r["min"] if a["min"] is None else min(a["min"], r["min"])
        a["max"] = r["max"] if a["max"] is None else max(a["max"], r["max"])
        a["sum"] += r["sum"]
        a["cnt"] += r["count"]
    out = {}
    for block, chunks in acc.items():
        out[block] = {}
        for chunk, a in chunks.items():
            mean = a["sum"] / a["cnt"] if a["cnt"] else float("nan")
            out[block][chunk] = (a["min"], mean, a["max"])
    return out


def plot_block(ax, series, title):
    chunks = sorted(series)
    mn = [series[c][0] for c in chunks]
    me = [series[c][1] for c in chunks]
    mx = [series[c][2] for c in chunks]
    ax.fill_between(chunks, mn, mx, alpha=0.22, color="tab:blue", label="min–max")
    ax.plot(chunks, me, color="tab:blue", lw=1.6, label="mean")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-1.0, 1.0)          # cosine similarity range
    ax.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax.grid(True, alpha=0.3)


def write_csv(agg, branch, out_csv):
    with open(out_csv, "w") as f:
        f.write("block,branch,chunk,min,mean,max\n")
        for block in sorted(agg):
            for chunk in sorted(agg[block]):
                mn, me, mx = agg[block][chunk]
                f.write(f"{block},{branch},{chunk},{mn:.6f},{me:.6f},{mx:.6f}\n")


def run_branch(records, branch, out_dir, per_block):
    agg = aggregate(records, branch)
    if not agg:
        print(f"[{branch}] no records, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    write_csv(agg, branch, os.path.join(out_dir, f"cluster_sim_{branch}.csv"))

    blocks = sorted(agg)
    # 1) Overview grid: one subplot per block.
    ncol = 5
    nrow = (len(blocks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for i, block in enumerate(blocks):
        ax = axes[i // ncol][i % ncol]
        plot_block(ax, agg[block], f"block {block}")
        if i % ncol == 0:
            ax.set_ylabel("top-1 cos-sim")
        if i // ncol == nrow - 1:
            ax.set_xlabel("chunk idx")
    for j in range(len(blocks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Top-1 cluster cos-sim vs chunk index — branch '{branch}'", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    grid_path = os.path.join(out_dir, f"blocks_grid_{branch}.png")
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f"[{branch}] grid -> {grid_path}")

    # 2) One separate figure per block (explicit request).
    if per_block:
        pb_dir = os.path.join(out_dir, f"per_block_{branch}")
        os.makedirs(pb_dir, exist_ok=True)
        for block in blocks:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_block(ax, agg[block], f"block {block} — branch '{branch}'")
            ax.set_xlabel("autoregressive chunk index")
            ax.set_ylabel("top-1 cosine similarity")
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(pb_dir, f"block_{block:02d}.png"), dpi=130)
            plt.close(fig)
        print(f"[{branch}] {len(blocks)} per-block figures -> {pb_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="cluster_sim_*.json file or a directory containing them")
    ap.add_argument("--branch", choices=["long", "short", "both"], default="long")
    ap.add_argument("--out", default=None, help="output dir (default: <input_dir>/plots)")
    ap.add_argument("--no-per-block", action="store_true", help="skip individual per-block figures")
    args = ap.parse_args()

    records, src = load_records(args.input)
    print(f"loaded {len(records)} records from {src}")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(src)), "plots")

    branches = ["long", "short"] if args.branch == "both" else [args.branch]
    for br in branches:
        run_branch(records, br, out_dir, per_block=not args.no_per_block)


if __name__ == "__main__":
    main()
