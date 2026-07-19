#!/bin/bash
# exp5: attenuate attention to RECENT tokens during the clean k/v prediction pass.
#
# During the clean-context pass (Step 2.3, which computes the clean k/v written to
# the cache and attended by future chunks), the query normally attends to
# [sink | mem | recent | curr]. This experiment multiplies the attention weight on
# the RECENT keys by SCALE (default 0.5 = halve) via an additive log(SCALE) logit
# bias, so the clean k/v rely less on recent context and more on sink/memory.
# Only the clean pass is affected; denoising passes are unchanged. Because the
# clean k/v are what get cached, this DOES change the generated video.
#
# Usage:
#   bash scripts/run_semantic_120s_exp5_clean_recent_scale.sh [scale] [config] [start] [end]
# Examples:
#   bash scripts/run_semantic_120s_exp5_clean_recent_scale.sh 0.5
#   bash scripts/run_semantic_120s_exp5_clean_recent_scale.sh 0.25 configs/selfforcing/semantic_120s.yaml 0 4
set -e

SCALE=${1:-0.5}
CONFIG=${2:-configs/selfforcing/semantic_120s.yaml}
START=${3:-4}
END=${4:-5}
OUT=./outputs/selfforcing/120s_semantic_exp5_clean_recent_scale

RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; c.model_kwargs.clean_recent_attn_scale=float('$SCALE'); OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (clean_recent_attn_scale=$SCALE) ==="
    python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Videos under: $OUT/"
