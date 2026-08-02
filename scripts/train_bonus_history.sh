#!/usr/bin/env bash
# Train and evaluate with models/bonus_history.yaml: exports the per-epoch
# metric history as JSON (--monitor-history-out) and renders loss, accuracy
# and precision curves together. Override the dataset with DATASET=<path> and
# the model path with MODEL_OUT=<path>.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="models/bonus_history.yaml"
TRAIN_FLAGS="--monitor-history-out reports/history_bonus_history_$(date +%Y%m%d-%H%M%S).json --monitor-metrics loss,accuracy,precision"

# shellcheck source=./_lib_train.sh
source "$(dirname "$0")/_lib_train.sh"
run_pipeline
