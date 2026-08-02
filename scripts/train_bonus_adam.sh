#!/usr/bin/env bash
# Train and evaluate with models/bonus_adam.yaml (Adam optimizer + weight
# decay). Override the dataset with DATASET=<path> and the model path with
# MODEL_OUT=<path>.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="models/bonus_adam.yaml"

# shellcheck source=./_lib_train.sh
source "$(dirname "$0")/_lib_train.sh"
run_pipeline
