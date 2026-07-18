#!/bin/bash
# exp3: memory-attention corrections. Runs generation with BOTH corrections on:
#   mem_logn_bias  — add log(n_i) to memory-key attention logits (n_i = per-prototype
#                    effective count; recovers attention mass of prototypes that
#                    absorbed many tokens). Uses the SDPA path (FA2 has no bias hook).
#   mem_key_renorm — rescale memory prototype key to running-mean raw norm r_i at
#                    read time (before RoPE), undoing EMA norm shrinkage.
# Both are model_kwargs booleans (default false). This script overrides them to true
# via a temp config (exp2 pattern) and writes to a separate output_folder, so the
# baseline (both off) config/outputs are untouched.
#
# Usage: bash scripts/run_semantic_120s_exp3_mem_logit.sh [config] [start] [end]
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-4}
OUT=./outputs/selfforcing/120s_semantic_exp3_key_norm

RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); c.output_folder='$OUT'; \
c.model_kwargs.mem_logn_bias=False; c.model_kwargs.mem_key_renorm=True; OmegaConf.save(c,'$RUN_CONFIG')"

echo "=== exp3 mem_logit: mem_logn_bias=ON, mem_key_renorm=ON -> $OUT ==="
python inference.py --config_path "$RUN_CONFIG" --start_idx "$START" --end_idx "$END"
echo "Done. Videos under: $OUT/"
