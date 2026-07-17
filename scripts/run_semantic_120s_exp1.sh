#!/bin/bash
# Run SemanticForcing inference with Self-Forcing base model (1 GPU, 40GB+ VRAM)
# Content-aware memory merging (cluster compression) variant of MemRoPE.
# Usage: bash scripts/run_semantic_120s.sh [config] [start_idx] [end_idx]

CONFIG=${1:-configs/selfforcing/semantic_120s.yaml}
START=${2:-0}
END=${3:-128}

echo "Config: $CONFIG"
echo "Prompts: $START to $END"

python inference.py \
    --config_path "$CONFIG" \
    --start_idx "$START" \
    --end_idx "$END"
