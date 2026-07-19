#!/bin/bash
# exp5 analysis (clean-kv attention overlay): the SAME per-slot attention-weight
# overlay as the v2 KEY_ATTEND_MAP analysis, but computed during the CLEAN-context
# pass (clean k/v prediction, causal_inference.py Step 2.3) instead of the
# denoising passes. For the chunk indices in CLEAN_KV_ATTEND, per (block) it
# records, per query token, the attention mass to each key slot
# (sink / mem_long / mem_short / recent / curr), renders a query x slot heatmap,
# and overlays the per-slot spatial attention onto the decoded RGB frames.
#
# No effect on the generated video (analysis-only recompute of the softmax).
#
# Usage:
#   bash scripts/run_semantic_120s_exp5_clean_kv_attend.sh [chunks] [config] [start] [end]
# Example:
#   bash scripts/run_semantic_120s_exp5_clean_kv_attend.sh "1,50,100,150"
set -e

CHUNKS=${1:-"90,95,100,115,135"}
CONFIG=${2:-configs/selfforcing/semantic_120s.yaml}
START=${3:-4}
END=${4:-5}
OUT=./outputs/selfforcing/120s_semantic_exp5_clean_kv_attend

RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (clean-pass attention overlay, chunks=$CHUNKS) ==="
    CLEAN_KV_ATTEND="$CHUNKS" python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    SAMPLE_DIR="$OUT/clean_kv_attend/$(printf 'sample_%04d' "$i")"
    if [ -d "$SAMPLE_DIR/spatial" ]; then
        python analysis/plot_key_attend_map.py "$SAMPLE_DIR"
    fi
    echo "  heatmaps : $SAMPLE_DIR/heatmap/   (query x slot per block, clean pass)"
    echo "  overlays : $SAMPLE_DIR/overlay/   (per-slot attention on RGB, chunks concat)"
done
echo "Done. Results under: $OUT/clean_kv_attend/sample_*/"
