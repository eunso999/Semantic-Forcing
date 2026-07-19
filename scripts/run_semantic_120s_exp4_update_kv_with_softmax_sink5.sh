#!/bin/bash
# exp5 (softmax/attention aggregation) with a LARGER sink: sink=5, recent=4, new=3.
#
# Identical to run_semantic_120s_exp4_update_kv_with_softmax.sh except sink_size is
# raised from 3 to 5. In eviction mode the recent window is the remainder of the
# fixed local window, so with local_attn_size=12 and num_frame_per_block=3:
#   recent = local_attn_size - sink - new = 12 - 5 - 3 = 4
# giving the attended layout [sink 5 | recent 4 | curr 3]. (Default script: sink 3,
# recent 6, new 3.) The KV cache size (12 frames) is unchanged.
#
# Usage:
#   bash scripts/run_semantic_120s_exp4_update_kv_with_softmax_sink5.sh [config] [start] [end] [gate] [tau] [beta] [alpha] [gate_fn] [ema_long] [blocks] [norm_restore] [temp]
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-4}
END=${3:-5}
GATE=${4:-matched} # matched or top1
TAU=${5:-0.6}
BETA=${6:-0.02}
ALPHA=${7:-0.5}
GATE_FN=${8:-sigmoid} # relu or sigmoid
EMA_LONG=${9:-0.0} # shadow long-memory EMA rate
BLOCKS=${10:-"all"} # which blocks to refine
NORM_RESTORE=${11:-true} # restore original head-norm after blend
TEMP=${12:-0.01} # attention temperature (smaller = sharper ~ top-1; larger = flatter ~ mean/blur)
SINK=5             # sink frames (this variant); recent auto = local_attn_size - sink - new
OUT=./outputs/selfforcing/120s_semantic_exp4_update_kv_with_softmax_sink5

# Config variant: eviction + shadow memory + value refinement (attn) + sink=5.
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
    echo "=== sample $i -> $OUT  (sink=$SINK recent=4 new=3; attn temp=$TEMP gate=$GATE alpha=$ALPHA) ==="
    python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Refined (softmax-attn, sink=5) videos under: $OUT/"
