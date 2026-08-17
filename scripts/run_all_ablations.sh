#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PREPARED_MANIFEST CHECKPOINT [CONFIG]" >&2
  exit 2
fi

manifest=$1
checkpoint=$2
config=${3:-configs/manuscript.yaml}

for ablation in B0 B1 B2 B3 B4 B5 B6 B7 B8 B9; do
  da-edgeformer evaluate \
    --config "$config" \
    --manifest "$manifest" \
    --checkpoint "$checkpoint" \
    --ablation "$ablation" \
    --output "outputs/ablations/$ablation"
done

