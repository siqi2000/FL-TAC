#!/usr/bin/env bash
set -e
# Single-card or multi-card: HF device_map="auto" automatically shards LLaMA-7B
# across all visible GPUs. Set CUDA_VISIBLE_DEVICES to control which.
#   single card:  CUDA_VISIBLE_DEVICES=0 bash scripts/run_dolly.sh
#   four  cards:  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_dolly.sh
python main.py --config configs/dolly_llama.yaml "$@"
