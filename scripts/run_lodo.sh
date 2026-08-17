#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PREPARED_ROOT [DEVICE]" >&2
  exit 2
fi

prepared_root=$1
device=${2:-auto}
datasets=(redd uk-dale refit enertalk)
seeds=(11 23 37 53 71)

for target in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    source_args=()
    calibration_args=()
    for source in "${datasets[@]}"; do
      if [[ "$source" != "$target" ]]; then
        source_args+=(--manifest "$prepared_root/$source/manifest.json")
        calibration_args+=(--calibration-manifest "$prepared_root/$source/manifest.json")
      fi
    done
    checkpoint="checkpoints/lodo/held-out-$target-seed-$seed.pt"
    da-edgeformer train-meta "${source_args[@]}" \
      --seed "$seed" --device "$device" --checkpoint "$checkpoint"
    da-edgeformer evaluate \
      --manifest "$prepared_root/$target/manifest.json" \
      "${calibration_args[@]}" \
      --checkpoint "$checkpoint" --ablation B9 --label-budget 0.05 \
      --seed "$seed" --device "$device" \
      --output "outputs/lodo/$target/seed-$seed"
  done
done
