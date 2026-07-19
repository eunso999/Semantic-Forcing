#!/bin/bash
# exp5 ("replace" mode): instead of soft-blending every clean recent value token
# toward memory, DELETE + FILL a subset. Tokens whose value poorly matches memory
# (value cos-sim vsim < TAU) are REPLACED by the memory prototype value; tokens
# that already match (vsim >= TAU) are left untouched.
#
# Implemented via gate_fn="hard": g = ALPHA * (vsim < TAU), reusing the same blend
# v = (1-g)*v + g*pv_match. ALPHA=1.0 => hard replace of selected tokens; ALPHA<1
# => partial blend of the selected. The memory target pv_match is the softmax
# attention read from memory (aggregate=attn), as in the sink5/softmax runs.
#
# Base = eviction + shadow memory + sink=5 (matches the current sink5 experiment).
#
# Usage:
#   bash scripts/run_semantic_120s_exp4_update_kv_replace.sh [config] [start] [end] [tau] [alpha] [temp] [sink] [blocks]
# Examples:
#   bash scripts/run_semantic_120s_exp4_update_kv_replace.sh                 # tau=0.6 alpha=1.0 (hard replace)
#   bash scripts/run_semantic_120s_exp4_update_kv_replace.sh configs/selfforcing/semantic_120s.yaml 4 5 0.6 0.5 0.01 5 all
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-128}
TAU=${4:-0.3}        # replace tokens with value cos-sim below this
ALPHA=${5:-1.0}      # 1.0 = hard replace; <1 = partial blend of selected tokens
TEMP=${6:-0.01}      # attention-read temperature (aggregate=attn)
SINK=${7:-5}         # sink frames (recent auto = local_attn_size - sink - new)
BLOCKS=${8:-"all"}
AGG=${9:-"top1"}       # memory value target: "attn" (softmax read) | "top1" (hard argmax match)
EMA_ALPHA=${10:-0.0}
OUT=./outputs/selfforcing/120s_semantic_exp4_update_kv_replace

RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "
from omegaconf import OmegaConf
c=OmegaConf.load('$CONFIG')
c.output_folder='$OUT'
c.model_kwargs.sink_size=$SINK
c.model_kwargs.compression_method='eviction'
c.model_kwargs.mem_side_buffer=True
c.model_kwargs.mem_value_refine=True
c.model_kwargs.mem_value_refine_gate='matched'
c.model_kwargs.mem_value_refine_gate_fn='hard'          # <-- replace (step gate)
c.model_kwargs.mem_value_refine_tau=float('$TAU')
c.model_kwargs.mem_value_refine_alpha=float('$ALPHA')
c.model_kwargs.mem_value_refine_aggregate='$AGG'
c.model_kwargs.mem_value_refine_temp=float('$TEMP')
c.model_kwargs.mem_value_refine_blocks='$BLOCKS'
c.model_kwargs.mem_value_refine_norm_restore=True
c.model_kwargs.ema_alpha_long=float('$EMA_ALPHA')
OmegaConf.save(c,'$RUN_CONFIG')
"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (REPLACE: vsim<$TAU -> pv_match, alpha=$ALPHA, sink=$SINK, temp=$TEMP) ==="
    python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Replace-refined videos under: $OUT/"
