#!/usr/bin/env bash
# Usage: bash scripts/run_experiment.sh configs/default.yaml
set -e

CONFIG=${1:-configs/default.yaml}

echo "Running experiment with config: $CONFIG"
python -m src.eval.run --config "$CONFIG"
