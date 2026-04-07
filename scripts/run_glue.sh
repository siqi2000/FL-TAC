#!/usr/bin/env bash
set -e
python main.py --config configs/glue_bert.yaml "$@"
