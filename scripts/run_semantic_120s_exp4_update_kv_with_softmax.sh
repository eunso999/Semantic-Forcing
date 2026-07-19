#!/bin/bash
# exp5 (softmax/attention aggregation): memory value refinement of clean recent
# tokens, but the memory value target is a SOFT per-head cosine ATTENTION read
# from memory instead of the hard top-1 match.
#
# Base = eviction + shadow long/short cluster memory (mem_side_buffer). At the
# clean-context pass, for each new clean recent token:
#   pv = per-head softmax( cos(new_k_h, proto_k_h) / temp ) @ proto_v_h   (attn read)
#   v  = (1-g)*v + g*pv,   g = alpha * gate((tau - vsim)/beta),  vsim = cos(v, pv)
# The KEY is unchanged; the refined value is stored to cache and attended by
# future chunks. The current frame's output is unchanged (clean-pass only).
#
# NOTE: this model's keys sit in a narrow cone, so a LARGE temp makes the
# attention near-uniform (averages all prototype values -> blur). Start with a
# SMALL temp (e.g. 0.02-0.1) and tune. temp->0 approaches the top-1 script.
#
# Usage:
#   bash scripts/run_semantic_120s_exp4_update_kv_with_softmax.sh [config] [start] [end] [gate] [tau] [beta] [alpha] [gate_fn] [ema_long] [blocks] [norm_restore] [temp]
# Examples:
#   bash scripts/run_semantic_120s_exp4_update_kv_with_softmax.sh
#   bash scripts/run_semantic_120s_exp4_update_kv_with_softmax.sh configs/selfforcing/semantic_120s.yaml 4 5 matched 0.5 0.05 0.5 sigmoid 0.005 all true 0.05
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-128}
GATE=${4:-matched} # matched or top1
TAU=${5:-0.6}
BETA=${6:-0.02}
ALPHA=${7:-0.5}
GATE_FN=${8:-sigmoid} # relu or sigmoid
EMA_LONG=${9:-0.0} # shadow long-memory EMA rate
BLOCKS=${10:-"all"} # which blocks to refine
NORM_RESTORE=${11:-true} # restore original head-norm after blend
TEMP=${12:-0.01} # attention temperature (smaller = sharper ~ top-1; larger = flatter ~ mean/blur)
OUT=./outputs/selfforcing/120s_semantic_exp4_update_kv_with_softmax

# Config variant: eviction + shadow memory + value refinement with ATTN aggregation.
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
c.model_kwargs.mem_value_refine_aggregate='attn'
c.model_kwargs.mem_value_refine_temp=float('$TEMP')
OmegaConf.save(c,'$RUN_CONFIG')
"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (eviction + shadow + value_refine AGGREGATE=attn temp=$TEMP gate=$GATE gate_fn=$GATE_FN tau=$TAU beta=$BETA alpha=$ALPHA ema_long=$EMA_LONG blocks=$BLOCKS) ==="
    python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Refined (softmax-attn) videos under: $OUT/"
