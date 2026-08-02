#!/usr/bin/env bash
# Train and evaluate with models/bonus_multiple_curves.yaml: monitors several
# metrics (loss, accuracy, precision) and renders each one's training and
# validation curve on a single stacked graph. Override the dataset with
# DATASET=<path> and the model path with MODEL_OUT=<path>.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="models/bonus_multiple_curves.yaml"
TRAIN_FLAGS="--monitor-metrics loss,accuracy,precision"

# shellcheck source=./_lib_train.sh
source "$(dirname "$0")/_lib_train.sh"
run_pipeline
