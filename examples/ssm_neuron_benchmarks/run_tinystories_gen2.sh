#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 RUN_NAME [extra-python-args]"
  exit 1
fi

RUN_NAME="$1"
shift || true

# Commit current changes with the run name
git add --all
git commit -m "$RUN_NAME" || true

# Run the TinyStories Gen2 benchmark and stream logs to stdout
python -u examples/ssm_neuron_benchmarks/tinystories_gen2.py "$RUN_NAME" "$@"


