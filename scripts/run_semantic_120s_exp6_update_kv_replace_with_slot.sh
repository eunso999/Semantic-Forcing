#!/bin/bash
# exp6 ("update slot"): split the eviction PAST window into two regions instead of
# refining every past frame. Layout (latent frames, window=12, block=3):
#
#   [sink 5 | update 3 (refined) | recent 1 (raw) | curr 3 (raw)]
#
# - recent (the single newest past frame) stays RAW.
# - update (the next 3 older past frames) hold REPLACE-refined values: when a frame
#   crosses recent->update on a roll, its value is refined ONCE (vsim<TAU tokens ->
#   memory prototype value, gate_fn="hard") and PERSISTED into cache["v"]. It is not
#   re-refined afterwards; the shadow memory still evicts the ORIGINAL (raw) value.
# - curr is written raw (exp5 curr-refine OFF: mem_value_refine=false).
#
# With num_frame_per_block=3 the roll is 3 frames, so update(3) turns over each roll
# and all 3 update frames are (re-)entered and refined once per roll. Base = eviction
# + shadow memory + sink=5 (matches the sink5 experiment; window/block unchanged).
#
# Usage:
#   bash scripts/run_semantic_120s_exp6_update_kv_replace_with_slot.sh [config] [start] [end] [tau] [alpha] [temp] [agg] [blocks] [update_size]
# Examples:
#   bash scripts/run_semantic_120s_exp6_update_kv_replace_with_slot.sh                 # tau=0.3 alpha=1.0 (hard replace), update_size=3
#   bash scripts/run_semantic_120s_exp6_update_kv_replace_with_slot.sh configs/selfforcing/semantic_120s.yaml 4 5 0.3 1.0 0.01 top1 all 3
set -e

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-128}
TAU=${4:-0.5}        # replace tokens with value cos-sim below this
ALPHA=${5:-1.0}      # 1.0 = hard replace; <1 = partial blend of selected tokens
TEMP=${6:-0.01}      # attention-read temperature (aggregate=attn)
AGG=${7:-top1}       # memory value target: "attn" (softmax read) | "top1" (hard argmax match)
BLOCKS=${8:-"all"}   # which transformer blocks refine the update slot
USIZE=${9:-3}        # update-region size in latent frames (recent = past - update)
SINK=5               # sink frames (recent auto = past - update)
OUT=./outputs/selfforcing/120s_semantic_exp6_update_kv_replace_with_slot

RUN_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$RUN_CONFIG"' EXIT
python -c "
from omegaconf import OmegaConf
c=OmegaConf.load('$CONFIG')
c.output_folder='$OUT'
c.model_kwargs.sink_size=$SINK
c.model_kwargs.compression_method='eviction'
c.model_kwargs.mem_side_buffer=True
c.model_kwargs.mem_value_refine=False                    # curr stays RAW (exp5 off)
c.model_kwargs.mem_value_refine_update_size=$USIZE       # <-- exp6 update slot
c.model_kwargs.mem_value_refine_gate='matched'
c.model_kwargs.mem_value_refine_gate_fn='hard'           # replace (step gate)
c.model_kwargs.mem_value_refine_tau=float('$TAU')
c.model_kwargs.mem_value_refine_alpha=float('$ALPHA')
c.model_kwargs.mem_value_refine_aggregate='$AGG'
c.model_kwargs.mem_value_refine_temp=float('$TEMP')
c.model_kwargs.mem_value_refine_blocks='$BLOCKS'
c.model_kwargs.mem_value_refine_norm_restore=True
OmegaConf.save(c,'$RUN_CONFIG')
"

for ((i=START; i<END; i++)); do
    echo "=== sample $i -> $OUT  (UPDATE-SLOT: [sink $SINK | update $USIZE(refined) | recent 1 | curr], vsim<$TAU -> pv_match[$AGG], alpha=$ALPHA) ==="
    python inference.py --config_path "$RUN_CONFIG" --start_idx "$i" --end_idx "$((i+1))"
done
echo "Done. Update-slot videos under: $OUT/"
