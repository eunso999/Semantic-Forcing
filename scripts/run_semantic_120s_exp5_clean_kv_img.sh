#!/bin/bash
# exp5 analysis (clean-kv image): compare, per chunk, the DENOISING output
# (Step 2.2 denoised_pred, what becomes the video) against the CLEAN-CONTEXT
# pass output (Step 2.3, causal_inference.py line ~205 — the model's x0
# prediction when re-run on the clean tokens at timestep=context_noise; normally
# discarded, only used to update the KV cache).
#
# With CLEAN_KV_IMG=1 the pipeline also decodes the clean-context output and
# inference.py saves a side-by-side video [denoising | clean-context] as
# *_cleancat.mp4. This visualizes how the clean pass would "re-refine" the clean
# tokens naturally — motivating a refinement built into clean-token generation
# rather than the forced prototype refinement of the earlier exp4/exp5 runs.
#
# No effect on the generated video itself (only an extra capture + decode).
#
# Usage:
#   bash scripts/run_semantic_120s_exp5_clean_kv_img.sh [config] [start] [end]
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-4}
END=${3:-5}
OUT=./outputs/selfforcing/120s_semantic_exp5_clean_kv_img

# Separate output dir; use the config's model settings as-is (no forced refine).
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; OmegaConf.save(c,'$RUN_CONFIG')"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (CLEAN_KV_IMG: denoised | clean-context side-by-side) ==="
    CLEAN_KV_IMG=1 python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Side-by-side videos: $OUT/rank*-*_*_cleancat.mp4"
