from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset

from da_edgeformer.audit import audit_protocol
from da_edgeformer.baselines.external import EXTERNAL_BASELINES, load_external_predictions
from da_edgeformer.checkpoints import load_model_checkpoint, save_checkpoint
from da_edgeformer.config import ExperimentConfig, load_config
from da_edgeformer.data.dataset import CausalWindowDataset, PreparedHousehold
from da_edgeformer.data.prepare import prepare_dataset
from da_edgeformer.evaluation.labels import deterministic_label_mask
from da_edgeformer.evaluation.metrics import household_macro_metrics
from da_edgeformer.evaluation.prequential import ABLATIONS, PrequentialEvaluator, PrequentialResult
from da_edgeformer.evaluation.schedules import matched_update_schedule
from da_edgeformer.evaluation.transitions import (
    detector_event_metrics,
    post_shift_mae_increase,
    recovery_delay,
)
from da_edgeformer.models.edgeformer import DAEdgeFormer
from da_edgeformer.profiling import profile_inference
from da_edgeformer.protocol import load_protocol, protocol_metadata
from da_edgeformer.release_audit import release_audit
from da_edgeformer.reporting import aggregate_summaries, discover_summaries, write_report
from da_edgeformer.reproducibility import set_reproducible_seed
from da_edgeformer.training.meta import FirstOrderMetaTrainer


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _manifest_households(
    manifest_path: str | Path, split: str, window_size: int
) -> list[CausalWindowDataset]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = []
    for entry in manifest["households"]:
        if entry["split"] == split:
            household = PreparedHousehold.load(manifest_path.parent / entry["path"])
            datasets.append(CausalWindowDataset(household, window_size))
    if not datasets:
        raise ValueError(f"manifest has no {split} households")
    return datasets


def _household_registry(
    manifest_paths: list[str], split: str, window_size: int
) -> dict[str, CausalWindowDataset]:
    registry: dict[str, CausalWindowDataset] = {}
    for manifest_path_value in manifest_paths:
        manifest_path = Path(manifest_path_value)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["households"]:
            if entry["split"] != split:
                continue
            household = PreparedHousehold.load(manifest_path.parent / entry["path"])
            key = f"{manifest['dataset']}:{household.household_id}"
            if key in registry:
                raise ValueError(f"duplicate household key: {key}")
            registry[key] = CausalWindowDataset(household, window_size)
    return registry


def _macro_result(result: object, datasets: list[CausalWindowDataset]) -> list[dict[str, object]]:
    household_ids = [
        f"{dataset.household.dataset}:{dataset.household.household_id}" for dataset in datasets
    ]
    result.metrics, records = household_macro_metrics(
        result.targets,
        result.predictions,
        result.target_states,
        result.predicted_states,
        result.target_mask,
        [len(dataset) for dataset in datasets],
        household_ids,
    )
    return records


def _save_evaluation(
    output_value: str,
    result: object,
    summary: dict[str, object],
    appliances: list[str],
) -> None:
    output = Path(output_value)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "predictions.npz",
        predictions=result.predictions,
        targets=result.targets,
        predicted_states=result.predicted_states,
        target_states=result.target_states,
        target_mask=result.target_mask,
        drift_scores=result.drift_scores,
        visible_labels=result.visible_labels,
        appliances=np.asarray(appliances),
    )


def _matched_schedule(
    args: argparse.Namespace,
    stream: ConcatDataset,
    stream_id: str,
    config: ExperimentConfig,
    trigger: str,
) -> tuple[set[int] | None, int | None]:
    source = getattr(args, "match_updates_from", None)
    if source is None:
        return None, None
    if trigger not in {"periodic", "random"}:
        raise ValueError("--match-updates-from is valid only for periodic or random triggers")
    reference = json.loads(Path(source).read_text(encoding="utf-8"))
    target_count = sum(event.get("decision") == "permitted" for event in reference["events"])
    fraction = config.training.label_budget if args.label_budget is None else args.label_budget
    label_mask = deterministic_label_mask(len(stream), fraction, args.seed, stream_id)
    timestamps = np.asarray([float(stream[index][4]) for index in range(len(stream))])
    return (
        matched_update_schedule(
            timestamps,
            label_mask,
            target_count,
            trigger,
            config.controller,
            args.seed,
            stream_id,
        ),
        target_count,
    )


def command_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    path = prepare_dataset(
        args.source_manifest,
        args.raw_root,
        args.output,
        config,
        protocol_metadata(protocol),
    )
    print(path)


def command_train_meta(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    fixed_sizes = protocol.values["training"]
    requested_sizes = {
        "support_windows": args.support_size,
        "query_windows": args.query_size,
        "retained_windows": args.retained_size,
    }
    for name, value in requested_sizes.items():
        if value != int(fixed_sizes[name]):
            raise ValueError(f"protocol fixes {name}={fixed_sizes[name]}, received {value}")
    if args.seed not in [int(value) for value in fixed_sizes["seeds"]]:
        raise ValueError(f"seed {args.seed} is not one of the protocol seeds")
    device = _device(args.device)
    set_reproducible_seed(args.seed)
    tasks = []
    validation_tasks = []
    for manifest in args.manifest:
        tasks.extend(_manifest_households(manifest, "train", config.model.window_size))
        validation_tasks.extend(
            _manifest_households(manifest, "validation", config.model.window_size)
        )
    model = DAEdgeFormer(config.model)
    trainer = FirstOrderMetaTrainer(model, config, device, args.seed)
    history = trainer.fit(
        tasks,
        validation_tasks=validation_tasks,
        support_size=args.support_size,
        query_size=args.query_size,
        retained_size=args.retained_size,
    )
    best_epoch = min(range(len(history)), key=lambda index: history[index]["validation_mae_w"])
    save_checkpoint(
        args.checkpoint,
        model,
        config,
        {
            "history": history,
            "best_epoch": best_epoch + 1,
            "seed": args.seed,
            **protocol_metadata(protocol),
        },
    )
    print(json.dumps({"checkpoint": args.checkpoint, "best_epoch": best_epoch + 1}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    device = _device(args.device)
    set_reproducible_seed(args.seed)
    model = DAEdgeFormer(config.model)
    load_model_checkpoint(args.checkpoint, model, device)
    calibration_manifests = args.calibration_manifest or [args.manifest]
    validation = []
    for manifest in calibration_manifests:
        validation.extend(_manifest_households(manifest, "validation", config.model.window_size))
    tests = _manifest_households(args.manifest, "test", config.model.window_size)
    policy = ABLATIONS[args.ablation]
    if args.trigger is not None:
        policy = replace(policy, trigger=args.trigger)
    evaluator = PrequentialEvaluator(model, config, policy, device, args.seed)
    if config.drift.threshold is None:
        threshold = evaluator.calibrate(validation)
    else:
        threshold = config.drift.threshold
    stream = ConcatDataset(tests)
    transitions = (
        {int(value) for value in args.oracle_transitions.split(",") if value.strip()}
        if args.oracle_transitions
        else set()
    )
    stream_id = args.stream_id or Path(args.manifest).stem
    schedule, matched_count = _matched_schedule(args, stream, stream_id, config, policy.trigger)
    result = evaluator.run(
        stream,
        stream_id,
        args.label_budget,
        transition_indices=transitions,
        scheduled_alarm_indices=schedule,
    )
    household_metrics = _macro_result(result, tests)
    summary = {
        "experiment": "single-dataset",
        "ablation": args.ablation,
        "trigger": policy.trigger,
        "seed": args.seed,
        "label_budget": args.label_budget,
        "drift_threshold": threshold,
        "metrics": result.metrics,
        "household_metrics": household_metrics,
        "events": result.events,
        "matched_update_count": matched_count,
        "matched_update_reference": Path(args.match_updates_from).name
        if args.match_updates_from
        else None,
        **protocol_metadata(protocol),
    }
    _save_evaluation(args.output, result, summary, config.model.appliances)
    print(json.dumps(summary["metrics"], indent=2))


def command_evaluate_natural(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    device = _device(args.device)
    set_reproducible_seed(args.seed)
    model = DAEdgeFormer(config.model)
    load_model_checkpoint(args.checkpoint, model, device)
    registry = _household_registry(args.manifest, "test", config.model.window_size)
    validation_registry = _household_registry(args.manifest, "validation", config.model.window_size)
    order_values = yaml.safe_load(Path(args.orders).read_text(encoding="utf-8"))["orders"]
    if not 1 <= args.order <= len(order_values):
        raise ValueError(f"order must be in [1, {len(order_values)}]")
    order = order_values[args.order - 1]
    missing = set(order) - set(registry)
    if missing:
        raise ValueError(f"stream order references missing prepared households: {sorted(missing)}")
    datasets = [registry[key] for key in order]
    policy = ABLATIONS[args.ablation]
    if args.trigger is not None:
        policy = replace(policy, trigger=args.trigger)
    evaluator = PrequentialEvaluator(model, config, policy, device, args.seed)
    threshold = (
        evaluator.calibrate(list(validation_registry.values()))
        if config.drift.threshold is None
        else config.drift.threshold
    )
    transitions = set(np.cumsum([len(dataset) for dataset in datasets])[:-1].tolist())
    stream = ConcatDataset(datasets)
    stream_id = f"natural-order-{args.order}"
    schedule, matched_count = _matched_schedule(args, stream, stream_id, config, policy.trigger)
    result = evaluator.run(
        stream,
        stream_id,
        args.label_budget,
        transition_indices=transitions if policy.trigger == "oracle" else set(),
        scheduled_alarm_indices=schedule,
    )
    household_metrics = _macro_result(result, datasets)
    alarms = [int(event["index"]) for event in result.events if bool(event["alarm"])]
    transition_config = protocol.values["metrics"]
    detector_metrics = detector_event_metrics(
        alarms,
        sorted(transitions),
        int(transition_config["detector_tolerance_samples"]),
        np.asarray([float(stream[index][4]) for index in range(len(stream))]),
    )
    absolute_errors = np.where(
        result.target_mask, np.abs(result.targets - result.predictions), np.nan
    )
    degradation = post_shift_mae_increase(
        absolute_errors,
        sorted(transitions),
        int(transition_config["recovery_reference_samples"]),
        int(transition_config["smoothing_block_samples"]),
    )
    recovery_delays = [
        recovery_delay(
            absolute_errors,
            transition,
            int(transition_config["recovery_reference_samples"]),
            int(transition_config["smoothing_block_samples"]),
            int(transition_config["recovery_consecutive_blocks"]),
            float(transition_config["recovery_tolerance"]),
        )
        for transition in sorted(transitions)
    ]
    summary = {
        "experiment": "natural-stream",
        "ablation": args.ablation,
        "trigger": policy.trigger,
        "seed": args.seed,
        "order": args.order,
        "household_order": order,
        "transition_indices": sorted(transitions),
        "label_budget": args.label_budget,
        "drift_threshold": threshold,
        "metrics": result.metrics,
        "household_metrics": household_metrics,
        "detector_metrics": detector_metrics,
        "post_shift_mae_increase": degradation,
        "recovery_delay_samples": recovery_delays,
        "events": result.events,
        "matched_update_count": matched_count,
        "matched_update_reference": Path(args.match_updates_from).name
        if args.match_updates_from
        else None,
        **protocol_metadata(protocol),
    }
    _save_evaluation(args.output, result, summary, config.model.appliances)
    print(json.dumps(summary["metrics"], indent=2))


def command_profile(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    device = _device(args.device)
    model = DAEdgeFormer(config.model)
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    model.freeze_backbone()
    profile = profile_inference(
        model,
        config.model.window_size,
        device,
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
    )
    print(json.dumps({**asdict(profile), **protocol_metadata(protocol)}, indent=2))


def command_evaluate_external(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    registry = _household_registry(args.manifest, "test", config.model.window_size)
    order_values = yaml.safe_load(Path(args.orders).read_text(encoding="utf-8"))["orders"]
    order = order_values[args.order - 1]
    missing = set(order) - set(registry)
    if missing:
        raise ValueError(f"stream order references missing prepared households: {sorted(missing)}")
    tests = [registry[key] for key in order]
    stream = ConcatDataset(tests)
    rows = [stream[index] for index in range(len(stream))]
    targets = np.stack([row[1].numpy() for row in rows])
    target_states = np.stack([row[2].numpy().astype(bool) for row in rows])
    target_mask = np.stack([row[3].numpy() for row in rows])
    baseline = EXTERNAL_BASELINES[args.baseline]
    stream_id = f"natural-order-{args.order}"
    visible_fraction = 0.05 if baseline.deployment_mode == "sparse_adaptation" else 0.0
    visible_labels = deterministic_label_mask(len(stream), visible_fraction, args.seed, stream_id)
    power, states = load_external_predictions(
        args.predictions,
        len(stream),
        config.model.appliances,
        visible_labels if visible_fraction else None,
    )
    result = PrequentialResult(
        metrics={},
        predictions=power,
        targets=targets,
        predicted_states=states,
        target_states=target_states,
        target_mask=target_mask,
        drift_scores=np.full(len(stream), np.nan),
        visible_labels=visible_labels,
    )
    household_metrics = _macro_result(result, tests)
    summary = {
        "experiment": "external-baseline",
        "baseline": baseline.name,
        "deployment_mode": baseline.deployment_mode,
        "visible_label_fraction": visible_fraction,
        "seed": args.seed,
        "order": args.order,
        "household_order": order,
        "metrics": result.metrics,
        "household_metrics": household_metrics,
        "events": [],
        **protocol_metadata(protocol),
    }
    _save_evaluation(args.output, result, summary, config.model.appliances)
    print(json.dumps(result.metrics, indent=2))


def command_smoke(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    set_reproducible_seed(args.seed)
    model = DAEdgeFormer(config.model)
    sample = torch.randn(4, config.model.window_size)
    output = model(sample)
    loss = output.power.mean() + output.state_logits.square().mean()
    loss.backward()
    print(
        json.dumps(
            {
                "power_shape": list(output.power.shape),
                "state_shape": list(output.state_logits.shape),
                "detector_shape": list(output.detector_features.shape),
                "parameters": model.parameter_counts(),
            },
            indent=2,
        )
    )


def command_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    protocol = load_protocol(args.protocol, config)
    report = audit_protocol(config, protocol, args.dataset_manifest, args.orders)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


def command_report(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    statistics = protocol.values["statistics"]
    summaries = discover_summaries(args.results)
    records = aggregate_summaries(
        summaries,
        iterations=int(statistics["bootstrap_iterations"]),
        seed=int(statistics["bootstrap_seed"]),
    )
    json_path, csv_path = write_report(records, args.output)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "groups": len(records)}))


def command_release_audit(args: argparse.Namespace) -> None:
    report = release_audit(args.root)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="da-edgeformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare user-supplied raw data")
    prepare.add_argument("--config", default="configs/manuscript.yaml")
    prepare.add_argument("--protocol", default="configs/protocol.yaml")
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--raw-root", required=True)
    prepare.add_argument("--output", default="prepared")
    prepare.set_defaults(function=command_prepare)

    train = subparsers.add_parser("train-meta", help="run household-task FOMAML")
    train.add_argument("--config", default="configs/manuscript.yaml")
    train.add_argument("--protocol", default="configs/protocol.yaml")
    train.add_argument("--manifest", action="append", required=True)
    train.add_argument("--checkpoint", default="checkpoints/meta.pt")
    train.add_argument("--support-size", type=int, default=256)
    train.add_argument("--query-size", type=int, default=256)
    train.add_argument("--retained-size", type=int, default=64)
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--device", default="auto")
    train.set_defaults(function=command_train_meta)

    evaluate = subparsers.add_parser("evaluate", help="prequential household-stream evaluation")
    evaluate.add_argument("--config", default="configs/manuscript.yaml")
    evaluate.add_argument("--protocol", default="configs/protocol.yaml")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument(
        "--calibration-manifest",
        action="append",
        help="validation manifest for detector calibration; repeatable (defaults to --manifest)",
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--ablation", choices=sorted(ABLATIONS), default="B9")
    evaluate.add_argument(
        "--trigger",
        choices=["none", "drift", "periodic", "random", "raw", "adwin", "oracle"],
        help="override the ablation's trigger for detector comparisons",
    )
    evaluate.add_argument(
        "--oracle-transitions",
        help="comma-separated stream indices; used only with the oracle trigger",
    )
    evaluate.add_argument("--label-budget", type=float, default=0.05)
    evaluate.add_argument("--stream-id")
    evaluate.add_argument(
        "--match-updates-from",
        help="B9 summary.json whose permitted-update count periodic/random must match",
    )
    evaluate.add_argument("--seed", type=int, default=11)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", default="outputs/evaluation")
    evaluate.set_defaults(function=command_evaluate)

    natural = subparsers.add_parser(
        "evaluate-natural", help="evaluate one predeclared cross-dataset household order"
    )
    natural.add_argument("--config", default="configs/manuscript.yaml")
    natural.add_argument("--protocol", default="configs/protocol.yaml")
    natural.add_argument("--orders", default="configs/stream_orders.yaml")
    natural.add_argument("--manifest", action="append", required=True)
    natural.add_argument("--checkpoint", required=True)
    natural.add_argument("--ablation", choices=sorted(ABLATIONS), default="B9")
    natural.add_argument(
        "--trigger",
        choices=["none", "drift", "periodic", "random", "raw", "adwin", "oracle"],
    )
    natural.add_argument("--label-budget", type=float, default=0.05)
    natural.add_argument("--order", type=int, choices=range(1, 11), required=True)
    natural.add_argument("--seed", type=int, default=11)
    natural.add_argument(
        "--match-updates-from",
        help="B9 summary.json whose permitted-update count periodic/random must match",
    )
    natural.add_argument("--device", default="auto")
    natural.add_argument("--output", default="outputs/natural")
    natural.set_defaults(function=command_evaluate_natural)

    profile = subparsers.add_parser("profile", help="profile causal inference")
    profile.add_argument("--config", default="configs/manuscript.yaml")
    profile.add_argument("--protocol", default="configs/protocol.yaml")
    profile.add_argument("--checkpoint")
    profile.add_argument("--device", default="cpu")
    profile.add_argument("--warmup", type=int, default=100)
    profile.add_argument("--iterations", type=int, default=1000)
    profile.add_argument("--threads", type=int, default=1)
    profile.set_defaults(function=command_profile)

    external = subparsers.add_parser(
        "evaluate-external", help="validate and score predictions from a controlled comparator"
    )
    external.add_argument("--config", default="configs/manuscript.yaml")
    external.add_argument("--protocol", default="configs/protocol.yaml")
    external.add_argument("--manifest", action="append", required=True)
    external.add_argument("--orders", default="configs/stream_orders.yaml")
    external.add_argument("--order", type=int, choices=range(1, 11), required=True)
    external.add_argument("--seed", type=int, default=11)
    external.add_argument("--baseline", choices=sorted(EXTERNAL_BASELINES), required=True)
    external.add_argument("--predictions", required=True)
    external.add_argument("--output", default="outputs/external")
    external.set_defaults(function=command_evaluate_external)

    smoke = subparsers.add_parser("smoke", help="run a data-free model smoke test")
    smoke.add_argument("--config", default="configs/smoke.yaml")
    smoke.add_argument("--seed", type=int, default=11)
    smoke.set_defaults(function=command_smoke)

    audit = subparsers.add_parser("audit", help="validate the fixed manuscript protocol")
    audit.add_argument("--config", default="configs/manuscript.yaml")
    audit.add_argument("--protocol", default="configs/protocol.yaml")
    audit.add_argument("--orders", default="configs/stream_orders.yaml")
    audit.add_argument(
        "--dataset-manifest",
        action="append",
        default=[
            "configs/datasets/manuscript/redd.yaml",
            "configs/datasets/manuscript/ukdale.yaml",
            "configs/datasets/manuscript/refit.yaml",
            "configs/datasets/manuscript/enertalk.yaml",
        ],
    )
    audit.set_defaults(function=command_audit)

    report = subparsers.add_parser(
        "report", help="aggregate household-level summaries with hierarchical intervals"
    )
    report.add_argument("--protocol", default="configs/protocol.yaml")
    report.add_argument("--results", default="outputs")
    report.add_argument("--output", default="reports")
    report.set_defaults(function=command_report)

    release = subparsers.add_parser(
        "release-audit", help="scan release-visible files for data, secrets, and unfinished markers"
    )
    release.add_argument("--root", default=".")
    release.set_defaults(function=command_release_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
