#!/bin/bash
# exp4: memory-blind attention (eviction) + shadow-memory cos-sim overlay.
#
# Generation is IDENTICAL to the wo_mem (eviction) run: compression_method='eviction',
# attention over [sink | recent | curr], no memory attended. In addition, with
# model_kwargs.mem_side_buffer=True, long/short cluster memory prototypes are
# maintained in a SIDE buffer (never attended, so the video is unchanged) using
# the same cluster_merge_update algorithm on eviction's evicted tokens.
#
# For the chunk indices in MEM_SIM_MAP, on the CLEAN-context pass we log, per new
# token, the top-1 cosine similarity of its clean key/value to the shadow long/
# short prototypes (4 slots: key_long, key_short, value_long, value_short) and
# overlay them onto the decoded RGB frames — one image per (block, slot), all
# analyzed chunks concatenated horizontally.
#
# Usage:
#   bash scripts/run_semantic_120s_exp4_analysis_cos_sim_map.sh [chunks] [config] [start] [end]
# Example:
#   bash scripts/run_semantic_120s_exp4_analysis_cos_sim_map.sh "90,95,100,115,135"
set -e

CHUNKS=${1:-"90,95,100,115,135"}
CONFIG=${2:-configs/selfforcing/semantic_120s.yaml}
START=${3:-4}
END=${4:-5}
OUT=./outputs/selfforcing/120s_semantic_exp4_analysis_cos_sim_map

# Config variant: separate output dir + eviction attention + shadow memory buffer.
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; c.model_kwargs.compression_method='eviction'; c.model_kwargs.mem_side_buffer=True; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (chunks=$CHUNKS, eviction + shadow memory) ==="
    MEM_SIM_MAP="$CHUNKS" python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    SAMPLE_DIR="$OUT/mem_sim_map/$(printf 'sample_%04d' "$i")"
    if [ -d "$SAMPLE_DIR/spatial" ]; then
        python analysis/plot_key_attend_map.py "$SAMPLE_DIR"
    fi
    echo "  overlays : $SAMPLE_DIR/overlay/   (block{b}_t0000_{key_long,key_short,value_long,value_short}.png)"
done
echo "Done. Results under: $OUT/mem_sim_map/sample_*/"
