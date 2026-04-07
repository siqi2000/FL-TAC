#!/usr/bin/env bash
set -e
python main.py --config configs/cifar_vit.yaml "$@"
