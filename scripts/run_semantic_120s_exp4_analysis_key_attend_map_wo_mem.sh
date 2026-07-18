#!/bin/bash
# Key-attention MAP analysis WITHOUT memory (exp3): same as the v2 analysis, but
# forces compression_method='eviction' so the KV cache is [sink | recent | curr]
# with NO long/short memory frames. Attention (and therefore the generated video)
# uses only [sink | recent | curr]; because eviction reclaims the 2 memory-frame
# slots for the recent window, recent is 2 frames wider than in cluster mode
# (recent = local_attn_size - sink - new = 6 vs 4).
#
# The v2 attention-map instrumentation is compression-method-agnostic: with no
# memory present the per-(block,timestep) heatmaps and RGB overlays are produced
# over the 3 slots [sink | recent | curr] (no mem_long/mem_short columns).
#
# Only the denoised_pred (denoising) passes are logged; the clean-context
# cache-update pass is not.
#
# Usage:
#   bash scripts/run_semantic_120s_exp4_analysis_key_attend_map_wo_mem.sh [chunks] [config] [start] [end]
# Examples:
#   bash scripts/run_semantic_120s_exp4_analysis_key_attend_map_wo_mem.sh "90,95,100,115,135"
set -e

CHUNKS=${1:-"90,95,100,115,135"}
CONFIG=${2:-configs/selfforcing/semantic_120s.yaml}
START=${3:-4}
END=${4:-5}
OUT=./outputs/selfforcing/120s_semantic_exp4_analysis_key_attend_map_wo_mem

# Config variant: separate output dir + compression_method forced to eviction
# (no memory -> attention over [sink | recent | curr]).
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; c.model_kwargs.compression_method='eviction'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (chunks=$CHUNKS, compression=eviction / no memory) ==="
    KEY_ATTEND_MAP="$CHUNKS" python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
    SAMPLE_DIR="$OUT/key_attend_map/$(printf 'sample_%04d' "$i")"
    if [ -d "$SAMPLE_DIR/spatial" ]; then
        python analysis/plot_key_attend_map.py "$SAMPLE_DIR"
    fi
    echo "  heatmaps : $SAMPLE_DIR/heatmap/   (part 2: query x [sink|recent|curr] per block,timestep)"
    echo "  overlays : $SAMPLE_DIR/overlay/   (part 3: slot attention on RGB, chunks concat)"
done
echo "Done. Results under: $OUT/key_attend_map/sample_*/"
