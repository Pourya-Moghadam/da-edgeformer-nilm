#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PREPARED_MANIFEST CHECKPOINT [CONFIG]" >&2
  exit 2
fi

manifest=$1
checkpoint=$2
config=${3:-configs/manuscript.yaml}

for budget in 0 0.005 0.01 0.05 0.10 1.0; do
  da-edgeformer evaluate \
    --config "$config" \
    --manifest "$manifest" \
    --checkpoint "$checkpoint" \
    --ablation B9 \
    --label-budget "$budget" \
    --output "outputs/label-budget/$budget"
done

