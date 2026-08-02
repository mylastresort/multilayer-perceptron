#!/usr/bin/env bash
# Shared split -> train -> predict pipeline used by the per-config train
# scripts. Must be sourced from a script that defines CONFIG (and optionally
# TRAIN_FLAGS) before sourcing. Dataset and model path are overridable via the
# DATASET and MODEL_OUT environment variables.
set -euo pipefail

run_pipeline() {
	local dataset="${DATASET:-data/data.csv}"
	local model_out="${MODEL_OUT:-models/model.json}"

	if [ ! -f "$dataset" ]; then
		echo "error: dataset not found: $dataset" >&2
		exit 1
	fi
	if [ ! -f "$CONFIG" ]; then
		echo "error: config not found: $CONFIG" >&2
		exit 1
	fi

	cargo build --quiet

	local train_csv="data/data_training.csv"
	local test_csv="data/data_test.csv"

	echo
	echo ">> Splitting dataset: $dataset"
	cargo run --quiet -- split --dataset "$dataset" --train-out "$train_csv" --val-out "$test_csv"

	echo
	echo ">> Training with config: $CONFIG"
	# shellcheck disable=SC2086
	TRAIN_OUTPUT=$(cargo run --quiet -- train --dataset "$train_csv" --config "$CONFIG" --model-out "$model_out" ${TRAIN_FLAGS:-} 2>&1)
	printf '%s\n' "$TRAIN_OUTPUT"

	echo
	echo ">> Evaluating saved model ($model_out) on the test split ($test_csv)"
	cargo run --quiet -- predict --dataset "$test_csv" --model "$model_out"

	echo
	echo ">> Learning curves written to:"
	printf '%s\n' "$TRAIN_OUTPUT" | grep "Learning curves saved:" | sed 's/^/   /'
}
