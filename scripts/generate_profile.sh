#!/bin/bash

# Generate a profiling report for data/data.csv using the same schema as EDA.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

base_features=(
  "Radius"
  "Texture"
  "Perimeter"
  "Area"
  "Smoothness"
  "Compactness"
  "Concavity"
  "Concave Points"
  "Symmetry"
  "Fractal Dimension"
)

stats=("mean" "se" "extreme")

names=("ID" "Diagnosis")
for feature in "${base_features[@]}"; do
  for stat in "${stats[@]}"; do
    names+=("${feature}_${stat}")
  done
done

joined_names=""
for i in "${!names[@]}"; do
  if [ "${i}" -gt 0 ]; then
    joined_names+=","
  fi
  joined_names+="${names[${i}]}"
done

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_pandas_profile.py" \
  "${ROOT_DIR}/data/data.csv" \
  --output-dir "${ROOT_DIR}/data" \
  --skiprows 1 \
  --header none \
  --names "${joined_names}"
