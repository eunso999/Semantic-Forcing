#!/bin/bash
# Key-attribution analysis: for each sample, log how much attention weight each
# query gives to the sink / mem_long / mem_short / recent / curr(new) key groups,
# then plot per-block "avg attention weight vs chunk index" as soon as it finishes.
# Usage: bash scripts/run_semantic_120s_exp2_analysis_key_attend_v1.sh [config] [start] [end]
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-10}
OUT=./outputs/selfforcing/120s_semantic_exp2_analysis_key_attend

# Config variant that writes to the separate analysis output dir.
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    NAME=$(printf 'key_attend_%04d' "$i")
    echo "=== sample $i -> $OUT ==="
    KEY_ATTEND_LOG=1 python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    JSON="$OUT/key_attend/$NAME.json"
    [ -f "$JSON" ] && python analysis/plot_key_attend.py "$JSON" --out "$OUT/key_attend/plots/$NAME"
done
echo "Done. Figures + tables under: $OUT/key_attend/plots/<sample>/"
