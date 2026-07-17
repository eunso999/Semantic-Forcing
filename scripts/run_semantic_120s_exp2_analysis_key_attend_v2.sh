#!/bin/bash
# Key-attention MAP analysis (v2): for a chosen set of autoregressive chunk
# indices, render (part 2) the full query x key attention heatmap per
# (block, timestep) and (part 3) overlay each key slot's attention onto the
# decoded RGB frames, concatenating all chunks horizontally.
#
# Only the denoised_pred (denoising) passes are logged; the clean-context
# cache-update pass is not.
#
# Usage:
#   bash scripts/run_semantic_120s_exp2_analysis_key_attend_v2.sh [chunks] [config] [start] [end]
# Examples:
#   bash scripts/run_semantic_120s_exp2_analysis_key_attend_v2.sh "1,50,100,150"
#   bash scripts/run_semantic_120s_exp2_analysis_key_attend_v2.sh "1,50,100,150" configs/selfforcing/semantic_120s.yaml 0 1
set -e

CHUNKS=${1:-"90,95,100,115,135"}
CONFIG=${2:-configs/selfforcing/semantic_120s.yaml}
START=${3:-4}
END=${4:-5}
OUT=./outputs/selfforcing/120s_semantic_exp2_analysis_key_attend_map

# Config variant that writes to the separate analysis output dir.
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (chunks=$CHUNKS) ==="
    KEY_ATTEND_MAP="$CHUNKS" python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    SAMPLE_DIR="$OUT/key_attend_map/$(printf 'sample_%04d' "$i")"
    if [ -d "$SAMPLE_DIR/spatial" ]; then
        python analysis/plot_key_attend_map.py "$SAMPLE_DIR"
    fi
    echo "  heatmaps : $SAMPLE_DIR/heatmap/   (part 2: query x key per block,timestep)"
    echo "  overlays : $SAMPLE_DIR/overlay/   (part 3: slot attention on RGB, chunks concat)"
done
echo "Done. Results under: $OUT/key_attend_map/sample_*/"
