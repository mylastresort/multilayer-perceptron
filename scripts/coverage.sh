#!/bin/bash

# Generates coverage reports using cargo-tarpaulin.
# - Html  → coverage/tarpaulin-report.html  (local viewing)
# - Lcov  → coverage/lcov.info              (uploaded to Codecov in CI)

set -e

echo "Running code coverage analysis..."

mkdir -p coverage

cargo tarpaulin \
  --out Html Lcov \
  --output-dir coverage \
  --jobs $(nproc) \
  --verbose

echo "Reports written to coverage/"
echo "  HTML : coverage/tarpaulin-report.html"
echo "  Lcov : coverage/lcov.info"
