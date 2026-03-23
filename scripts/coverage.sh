#!/bin/bash

# Generates HTML coverage report using cargo-tarpaulin
# Output goes to coverage/ directory

set -e

echo "Running code coverage analysis..."

mkdir -p coverage

cargo tarpaulin --out Html --output-dir coverage --verbose
