#!/bin/bash
# Key-attribution analysis WITHOUT content-aware clustering: same as
# run_semantic_120s_exp2_analysis_key_attend_v1.sh but forces the compression
# method to plain 'ema' (fixed (h,w)-bin EMA memory) instead of 'cluster'
# (content prototypes). Everything else — the sink/mem_long/mem_short/recent/
# curr key-group attribution, per-timestep logging, and plots — is identical,
# so the two runs are directly comparable (cluster vs. ema memory).
# Usage: bash scripts/run_semantic_120s_exp2_analysis_key_attend_v1_wo_cluster.sh [config] [start] [end]
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-10}
OUT=./outputs/selfforcing/120s_semantic_exp2_analysis_key_attend_wo_cluster

# Config variant: separate output dir + compression_method forced to 'ema'.
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; c.model_kwargs.compression_method='ema'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    NAME=$(printf 'key_attend_%04d' "$i")
    echo "=== sample $i -> $OUT  (compression=ema) ==="
    KEY_ATTEND_LOG=1 python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    JSON="$OUT/key_attend/$NAME.json"
    [ -f "$JSON" ] && python analysis/plot_key_attend.py "$JSON" --out "$OUT/key_attend/plots/$NAME"
done
echo "Done (ema, no cluster). Figures + tables under: $OUT/key_attend/plots/<sample>/"
