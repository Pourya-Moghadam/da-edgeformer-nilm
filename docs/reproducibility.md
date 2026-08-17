# Manuscript-to-code map

| Manuscript component | Implementation | Primary configuration |
|---|---|---|
| Causal multi-scale convolution | `models/layers.py` | `conv_kernels`, `conv_dilations` |
| Local causal attention | `LocalCausalSelfAttention` | `attention_heads`, `local_span` |
| Residual bottleneck adapters | `ResidualAdapter` | `adapter_bottleneck` |
| Power/state objective | `losses.prediction_loss` | `huber_delta_w`, `classification_weight` |
| Sparse-label FOMAML | `training/meta.py` | inner/outer rates and steps |
| Frozen-space mean-shift score | `adaptation/drift.py` | EWMA betas, quantile, consecutive count |
| Replay and EWC approximation | `adaptation/replay.py`, `adaptation/ewc.py` | replay/stability weights |
| Resource gate | `adaptation/controller.py` | capacity, refill, cooldown, minimum labels |
| Prediction-first deployment | `evaluation/prequential.py` | label budget and seed |
| MAE/NMAE/SAE/F1/MCC | `evaluation/metrics.py` | appliance order/thresholds |
| Household uncertainty | `evaluation/statistics.py` | bootstrap/permutation seed |
| Edge inference profile | `profiling.py` | warmup, iterations, threads |

## Internal ablations

The `B0`–`B9` names are accepted by `da-edgeformer evaluate`. The runtime policy controls
full versus adapter updates, periodic versus drift triggers, replay, and stability. B8 and
B9 additionally assume a meta-trained checkpoint; the evaluator cannot infer checkpoint
provenance, so that provenance is stored when `train-meta` writes the checkpoint.

Periodic and random comparisons use `--match-updates-from` with the B9 `summary.json` for
the same seed and stream order. The evaluator selects exactly that many alarms from the
gate-feasible schedule: evenly spaced for periodic and without replacement for seeded
random. The full manuscript script applies this rule automatically and records the count
and reference in each summary.

Trigger comparisons can override the selected ablation with `--trigger`. Supported modes
are `periodic`, `random`, `raw`, `adwin`, `drift`, and evaluator-only `oracle`. For example:

```bash
da-edgeformer evaluate ... --ablation B9 --trigger adwin
da-edgeformer evaluate ... --ablation B9 --trigger oracle \
  --oracle-transitions 1200,8400,15700
```

`adaptation/adwin.py` is a compact, version-controlled implementation; its defaults are
part of this repository's protocol and should not be assumed identical to another ADWIN
library. `evaluation/transitions.py` computes event precision/recall/F1, detection delay,
false alarms per day, missed transitions, post-shift degradation, and recovery delay.

## Natural streams and leave-one-dataset-out

A prepared dataset manifest has train, validation, and test households. `evaluate`
concatenates test households in manifest order. `evaluate-natural` loads the ten exact
cross-dataset orders from `configs/stream_orders.yaml` and assigns a unique stream ID.

For leave-one-dataset-out (LODO), meta-train a fresh checkpoint on the three source dataset
manifests, calibrate the detector on those source validation households with repeated
`--calibration-manifest`, and evaluate the held-out dataset. Rotate this process four times.
Do not use the held-out validation households or reuse a four-dataset checkpoint for LODO.

## Leakage controls

- normalization is calculated from training households only;
- detector thresholds are calibrated from validation streams only;
- causal windows exclude aggregate gaps; missing appliance-household pairs and channel gaps
  remain present with an explicit target mask and never enter a loss or metric;
- labels are revealed after the corresponding prediction is saved;
- label selection is deterministic for `(seed, stream-id, fraction)`;
- source and prepared-file checksums are stored in the preparation manifest;
- chronological support/query episodes have a purge of at least `W - 1` samples.

## Result artifacts

Each evaluation produces:

- `summary.json`: configuration identity, threshold, metrics, and every alarm/gate/update;
- `predictions.npz`: predictions, evaluator targets, states, detector scores, label mask,
  and appliance order.

These files are ignored by default because they can encode household routines. Publish
only reviewed aggregate results unless dataset terms and the privacy plan permit more.
