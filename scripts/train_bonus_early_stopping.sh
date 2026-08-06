#!/usr/bin/env bash
# Train and evaluate with models/bonus_early_stopping.yaml: enables early
# stopping (--early-stopping) and rolls the model back to the best monitored
# epoch. Override the dataset with DATASET=<path> and the model path with
# MODEL_OUT=<path>.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="models/bonus_early_stopping.yaml"

TRAIN_FLAGS="--early-stopping"

# shellcheck source=./_lib_train.sh
source "$(dirname "$0")/_lib_train.sh"
run_pipeline
