#!/bin/bash
# Cluster-similarity novelty analysis: run each sample, then plot its per-block
# top-1 cosine-similarity-vs-chunk graph as soon as that sample finishes.
# Usage: bash scripts/run_semantic_120s_exp2_analysis_novelty.sh [config] [start] [end] [branch]
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-10}
BRANCH=${4:-both}   # long | short | both
OUT=./outputs/selfforcing/120s_semantic_exp2_analysis_novelty

# Config variant that writes to the separate analysis output dir.
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    NAME=$(printf 'cluster_sim_%04d' "$i")
    echo "=== sample $i -> $OUT ==="
    CLUSTER_SIM_LOG=1 python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    JSON="$OUT/cluster_sim/$NAME.json"
    [ -f "$JSON" ] && python analysis/plot_cluster_sim.py "$JSON" --branch "$BRANCH" --out "$OUT/cluster_sim/plots/$NAME"
done
echo "Done. Figures under: $OUT/cluster_sim/plots/<sample>/"
