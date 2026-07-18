#!/bin/bash
# exp5: memory-prototype VALUE refinement of clean recent tokens.
#
# Base generation = eviction (attention over [sink | recent | curr], memory not
# attended) + shadow long/short cluster memory (mem_side_buffer). ON TOP, at the
# clean-context pass, each new clean recent VALUE token is refined toward its
# key-matched long-memory prototype's value via a soft-gated convex blend:
#   v = (1-g)*v + g*proto_v[j*],  g = alpha * sigmoid((tau - vsim)/beta)
# where j* = argmax key cos-sim, and vsim = value cos-sim (gate="matched": to the
# key-matched proto value; gate="top1": best value match). Low value cos-sim
# (likely artifact) -> larger g. The KEY is left unchanged; the refined value is
# what gets stored to the cache and attended by future chunks. The current
# frame's output is unchanged (refinement is clean-pass only).
#
# Usage:
#   bash scripts/run_semantic_120s_exp4_update_kv_prototype.sh [config] [start] [end] [gate] [tau] [beta] [alpha] [gate_fn]
# Examples:
#   bash scripts/run_semantic_120s_exp4_update_kv_prototype.sh
#   bash scripts/run_semantic_120s_exp4_update_kv_prototype.sh configs/selfforcing/semantic_120s.yaml 4 5 matched 0.6 0.1 1.0 sigmoid
#   bash scripts/run_semantic_120s_exp4_update_kv_prototype.sh configs/selfforcing/semantic_120s.yaml 4 5 matched 0.6 0.1 1.0 relu
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-4}
END=${3:-5}
GATE=${4:-matched} # matched or top1
TAU=${5:-0.5}
BETA=${6:-0.05} # 0.1
ALPHA=${7:-0.5}
GATE_FN=${8:-sigmoid} # relu or sigmoid
EMA_LONG=${9:-0.005} # shadow long-memory EMA rate (smaller = slower/less update; config default was 0.01)
BLOCKS=${10:-"all"} # which blocks to refine: "all" | "10,11,12" | "10-20" | "0,5,10-12"
NORM_RESTORE=${11:-true} # restore original head-norm after blend (avoid blur): true|false
OUT=./outputs/selfforcing/120s_semantic_exp4_update_kv_prototype

# Config variant: eviction attention + shadow memory + value refinement.
RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "
from omegaconf import OmegaConf
c=OmegaConf.load('$CONFIG')
c.output_folder='$OUT'
c.model_kwargs.compression_method='eviction'
c.model_kwargs.mem_side_buffer=True
c.model_kwargs.mem_value_refine=True
c.model_kwargs.mem_value_refine_gate='$GATE'
c.model_kwargs.mem_value_refine_tau=float('$TAU')
c.model_kwargs.mem_value_refine_beta=float('$BETA')
c.model_kwargs.mem_value_refine_alpha=float('$ALPHA')
c.model_kwargs.mem_value_refine_gate_fn='$GATE_FN'
c.model_kwargs.ema_alpha_long=float('$EMA_LONG')
c.model_kwargs.mem_value_refine_blocks='$BLOCKS'
c.model_kwargs.mem_value_refine_norm_restore=('$NORM_RESTORE'.lower() in ('true','1','yes'))
OmegaConf.save(c,'$RUN_CONFIG')
"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (eviction + shadow + value_refine gate=$GATE gate_fn=$GATE_FN tau=$TAU beta=$BETA alpha=$ALPHA ema_long=$EMA_LONG) ==="
    python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Refined videos under: $OUT/"
