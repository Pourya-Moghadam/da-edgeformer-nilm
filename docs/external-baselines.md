# Controlled external baselines

The manuscript compares STNILM, AugLPN-NILM, Energformer, ConvTransNILM, Metric Meta-NILM,
and RTNILM. `configs/baselines.yaml` records the implementation provenance used for each
controlled rerun. STNILM and Energformer use the pinned MIT-licensed NILMFormer
reimplementations. AugLPN-NILM uses the paper-linked author repository in place because
that repository declares no software license. No public source was found for the remaining
three; their reruns are paper-derived external implementations and cannot be redistributed
or represented as author-official code.

Fetch and verify the three public implementations without adding them to this repository:

```bash
bash scripts/fetch_public_baselines.sh
```

The ignored `external/` directory is checked out at the exact commits recorded in
`configs/baselines.yaml`.

For a controlled comparison:

1. Use the exact prepared manifest produced by this repository.
2. Feed the same normalized 256-sample causal windows to the upstream implementation.
3. Preserve test order and never expose evaluator-only labels to a static baseline.
4. For Metric Meta-NILM, use the identical post-prediction label-mask array.
5. Export predictions as an NPZ containing:
   - `power`: float array `[samples, appliances]` in watts;
   - `state`: bool array `[samples, appliances]`;
   - `appliances`: string array in manifest order.
   - `visible_labels`: required for Metric Meta-NILM and exactly equal to this repository's
     deterministic post-prediction label mask.
6. Validate and score the artifact with `da-edgeformer evaluate-external --baseline NAME
   --manifest MANIFEST --predictions FILE --output DIRECTORY`.
7. Profile all models with the same input, precision, thread count, warmup, measurement
   count, device state, and profiler convention.

The names and deployment modes are recorded in `baselines/external.py`. The validation
contract is complete, but the three non-public implementations are a reproducibility
limitation of the comparator study. A number from an upstream paper is not a controlled
rerun and must not be inserted into generated comparison tables unless its protocol is
demonstrably identical.
