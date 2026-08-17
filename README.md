# DA-EdgeFormer NILM

Reference implementation for **Drift-Aware, Parameter-Efficient Continual NILM under
Sparse Labels and Edge Budgets**.

DA-EdgeFormer performs causal non-intrusive load monitoring (NILM) with a frozen
convolution-attention backbone, small online-trainable adapters, post-prediction sparse
labels, stable-feature drift monitoring, bounded replay, an EWC-style stability penalty,
and token-bucket update control.

> **No dataset is distributed with this repository.** Obtain each dataset from its
> provider, accept its terms, and pass its local path to the preparation command. Raw
> data, prepared arrays, checkpoints, and outputs are ignored by Git.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/manuscript-cpu.txt
python -m pip install -e '.[dev]' --no-deps
da-edgeformer smoke
```

The reference CPU dependencies are exactly pinned. CI uses the smaller pins in
`requirements/ci-cpu.txt` and then installs this package with `--no-deps`. CUDA and
Raspberry Pi installations may require a platform-specific PyTorch wheel; record any
deviation alongside the generated profile.

The smoke command constructs the model, runs a forward/backward pass on random tensors,
and does not download or write data.

## Obtain the datasets

- [REDD description and original resource links](https://bigdata.seas.gwu.edu/data-set-36-the-reference-energy-disaggregation-data-set/)
- [UK-DALE article and data-format description](https://www.nature.com/articles/sdata20157)
- [REFIT provider record and downloads](https://pureportal.strath.ac.uk/en/datasets/31da3ece-f902-4e95-a093-e0a9536983c4)
- [ENERTALK article and data record](https://www.nature.com/articles/s41597-019-0212-5)

Availability and terms are controlled by the providers. In particular, mirrors can
contain altered or preprocessed copies; record the source and checksum in your own
experiment notes.

## Prepare user-supplied data

Preparation is driven by a fixed manuscript source manifest. It records household IDs, source paths,
aggregate channels, appliance channels, and the household-disjoint split. Two input
formats are supported:

- `channel_dat`: REDD/UK-DALE directories containing `channel_N.dat` and `labels.dat`.
- `csv`: native cleaned REFIT CSV files with metadata-backed column mappings.
- `parquet_directory`: native ENERTALK house/date/device Parquet directories.

```bash
da-edgeformer prepare \
  --config configs/manuscript.yaml \
  --source-manifest configs/datasets/manuscript/redd.yaml \
  --raw-root /path/to/REDD \
  --output prepared
```

The command:

1. reads the source in place;
2. computes time-weighted means on a 10-second grid;
3. leaves long gaps missing rather than interpolating them;
4. derives state labels using manifest-recorded thresholds;
5. computes normalization from training households only; and
6. writes one compressed array per household plus `manifest.json` with checksums.

Prepared files remain local under the ignored `prepared/` directory.

## Train and evaluate

Meta-train on one or more prepared manifests:

```bash
da-edgeformer train-meta \
  --config configs/manuscript.yaml \
  --manifest prepared/redd/manifest.json \
  --manifest prepared/uk-dale/manifest.json \
  --manifest prepared/refit/manifest.json \
  --manifest prepared/enertalk/manifest.json \
  --seed 11 \
  --checkpoint checkpoints/meta-seed11.pt
```

Run strict prequential evaluation. The evaluator predicts first, records the evaluator-only
target, reveals only the deterministic label mask, updates detector/buffers, and finally
checks whether adaptation is permitted.

```bash
da-edgeformer evaluate \
  --config configs/manuscript.yaml \
  --manifest prepared/refit/manifest.json \
  --checkpoint checkpoints/meta-seed11.pt \
  --ablation B9 \
  --label-budget 0.05 \
  --seed 11 \
  --output outputs/refit/B9/seed11
```

If `drift.threshold` is `null`, the detector threshold is calibrated from validation
households before the test stream. It never uses test labels for calibration. Run the
complete internal ablation with:

```bash
bash scripts/run_all_ablations.sh \
  prepared/refit/manifest.json checkpoints/meta-seed11.pt
```

Profile the same 256-sample FP32 input with a single CPU thread:

```bash
da-edgeformer profile \
  --config configs/manuscript.yaml \
  --checkpoint checkpoints/meta-seed11.pt \
  --device cpu --threads 1 --warmup 100 --iterations 1000
```

Raw timing samples are returned alongside p50/p95 latency and peak process RSS. CPU
frequency, temperature, OS image, PyTorch build, and power traces must be recorded
separately for a defensible edge benchmark.

## Reproducibility protocol

Every value explicitly stated in the manuscript is used directly. Every unstated choice is
fixed in `configs/protocol.yaml`, including dataset ordering, appliance priority, missing
target masks, initialization, optimizer details, statistical iterations, and profiling
conventions. The ten natural stream orders are materialized in `configs/stream_orders.yaml`.
These files are fixed, versioned protocol inputs for the manuscript runs.

The stated width and adapter bottleneck do not alone determine the reported parameter
counts. The protocol therefore fixes a 372-channel multi-scale convolutional expansion and
516-unit appliance heads. The resulting code-generated model is approximately 0.64 M total
parameters with approximately 0.074 M online-trainable parameters. No unused padding
parameters are present.

See [docs/reproducibility.md](docs/reproducibility.md) for the table/experiment mapping and
[docs/external-baselines.md](docs/external-baselines.md) for controlled comparator imports.

After preparing all four datasets, the complete five-seed, ten-order experiment matrix is:

```bash
bash scripts/run_manuscript.sh prepared auto
bash scripts/run_lodo.sh prepared auto
da-edgeformer report --results outputs --output reports
```

The complete run is computationally substantial. `summary.json` files retain every alarm,
gate decision, and update; `da-edgeformer report` verifies a single protocol identity and
produces household-hierarchical aggregate JSON and CSV files. Run `da-edgeformer audit`
to validate the fixed manuscript protocol and `da-edgeformer release-audit` before sharing
the repository.

## Testing

```bash
ruff check .
pytest
```

Tests cover causal invariance, local attention, gap-safe resampling, deterministic label
visibility, token/cooldown gating, metrics, configuration validation, and a synthetic
prequential run.

## License and citation

Code is released under the [MIT License](LICENSE). Dataset licenses are separate and are
not granted by this repository. Citation metadata is available in [CITATION.cff](CITATION.cff).
