#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PREPARED_ROOT [DEVICE]" >&2
  exit 2
fi

prepared_root=$1
device=${2:-auto}
config=configs/manuscript.yaml
protocol=configs/protocol.yaml
manifests=(
  "$prepared_root/redd/manifest.json"
  "$prepared_root/uk-dale/manifest.json"
  "$prepared_root/refit/manifest.json"
  "$prepared_root/enertalk/manifest.json"
)
manifest_args=()
for manifest in "${manifests[@]}"; do
  manifest_args+=(--manifest "$manifest")
done

seeds=(11 23 37 53 71)
for seed in "${seeds[@]}"; do
  checkpoint="checkpoints/manuscript/seed-$seed.pt"
  da-edgeformer train-meta \
    --config "$config" --protocol "$protocol" \
    "${manifest_args[@]}" --seed "$seed" --device "$device" \
    --checkpoint "$checkpoint"

  for order in {1..10}; do
    # B9 is evaluated first because matched periodic/random schedules use its
    # realized permitted-update count for this exact seed and stream order.
    for ablation in B9 B0 B1 B2 B3 B4 B5 B6 B7 B8; do
      da-edgeformer evaluate-natural \
        --config "$config" --protocol "$protocol" \
        "${manifest_args[@]}" --checkpoint "$checkpoint" \
        --seed "$seed" --order "$order" --ablation "$ablation" \
        --label-budget 0.05 --device "$device" \
        --output "outputs/natural/seed-$seed/order-$order/$ablation"
    done
  done

  for order in {1..10}; do
    for budget in 0 0.005 0.01 0.05 0.10 1.0; do
      da-edgeformer evaluate-natural \
        --config "$config" --protocol "$protocol" \
        "${manifest_args[@]}" --checkpoint "$checkpoint" \
        --seed "$seed" --order "$order" --ablation B9 --label-budget "$budget" \
        --device "$device" \
        --output "outputs/label-budget/seed-$seed/order-$order/$budget"
    done

    reference="outputs/natural/seed-$seed/order-$order/B9/summary.json"
    for trigger in periodic random raw adwin drift oracle; do
      match_args=()
      if [[ "$trigger" == periodic || "$trigger" == random ]]; then
        match_args=(--match-updates-from "$reference")
      fi
      da-edgeformer evaluate-natural \
        --config "$config" --protocol "$protocol" \
        "${manifest_args[@]}" --checkpoint "$checkpoint" \
        --seed "$seed" --order "$order" --ablation B9 --trigger "$trigger" \
        --label-budget 0.05 --device "$device" "${match_args[@]}" \
        --output "outputs/triggers/seed-$seed/order-$order/$trigger"
    done
  done
done
