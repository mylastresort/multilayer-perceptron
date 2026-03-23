#!/bin/bash

# Updates the coverage badge in README.md with the latest tarpaulin output.
#
# Expects README.md to contain a badge in this format:
#   <img src="https://img.shields.io/badge/coverage-0.0%25-orange" alt="Coverage">
#
# The script replaces the percentage and colour automatically:
#   >= 80%  → green
#   >= 60%  → yellow
#   <  60%  → orange

set -e

echo "Generating coverage data for README update..."

# Run tarpaulin and capture stdout output
COVERAGE_OUTPUT=$(cargo tarpaulin --out Stdout --jobs $(nproc) 2>/dev/null || echo "")

if [ -z "$COVERAGE_OUTPUT" ]; then
  echo "Failed to generate coverage data"
  exit 1
fi

# Extract overall coverage percentage (e.g. "87.50%")
OVERALL_LINE=$(echo "$COVERAGE_OUTPUT" | grep "% coverage" | tail -1)
OVERALL_COVERAGE=$(echo "$OVERALL_LINE" | sed -n 's/^\([0-9]\+\.[0-9]\+%\) coverage.*/\1/p')
OVERALL_LINES=$(echo "$OVERALL_LINE" | sed -n 's/.* \([0-9]\+\/[0-9]\+\) lines covered.*/\1/p')

if [ -z "$OVERALL_COVERAGE" ]; then
  echo "Could not parse coverage percentage from tarpaulin output"
  exit 1
fi

echo "Overall coverage: $OVERALL_COVERAGE ($OVERALL_LINES lines)"

# Pick badge colour based on coverage value
NUMERIC=$(echo "$OVERALL_COVERAGE" | sed 's/%//')
if awk "BEGIN {exit !($NUMERIC >= 80)}"; then
  COLOUR="green"
elif awk "BEGIN {exit !($NUMERIC >= 60)}"; then
  COLOUR="yellow"
else
  COLOUR="orange"
fi

# URL-encode the % sign for the shields.io URL
COVERAGE_ENCODED=$(echo "$OVERALL_COVERAGE" | sed 's/%/%25/g')

# Replace the badge line in README.md
sed -i "s|https://img\.shields\.io/badge/coverage-[0-9]\+\.[0-9]\+%25-[a-z]\+|https://img.shields.io/badge/coverage-${COVERAGE_ENCODED}-${COLOUR}|g" README.md

echo "README.md updated — coverage badge set to ${OVERALL_COVERAGE} (${COLOUR})"
