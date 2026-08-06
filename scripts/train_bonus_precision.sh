#!/usr/bin/env bash
# Train and evaluate with models/bonus_precision.yaml: precision is the
# early-stopping metric (--early-stop-metric precision) while loss, accuracy
# and precision curves are still rendered. Override the dataset with
# DATASET=<path> and the model path with MODEL_OUT=<path>.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="models/bonus_precision.yaml"
TRAIN_FLAGS="--early-stopping --early-stop-metric precision --early-stop-mode max --monitor-metrics loss,accuracy,precision"

# shellcheck source=./_lib_train.sh
source "$(dirname "$0")/_lib_train.sh"
run_pipeline
