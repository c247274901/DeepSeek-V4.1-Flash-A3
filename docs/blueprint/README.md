# DeepSeek V4.1 Flash A3 task catalog

Blueprint can read each model change against its parent. The history index
records the source SHA, parent and touched files for each commit, including
reverted attempts. The 1122 branch was restarted at 53ea3e6. Its model files
match the earlier 13f75e54 snapshot, so the archive places that change
after the original graph flattening. [History index](history.json)

## Architecture baselines

- baseline/architecture-before captures staged Prefill and resident Decode.
- baseline/full-graph-entry adds the integrated Prefill and Decode programs.
- baseline/pre-flatten captures the subsequent shared kernel and host changes.
- baseline/architecture-after captures the flattened graph.
- release/1122-kernels reproduces the selected upstream model snapshot.

The architecture comparisons use architecture-before → full-graph-entry
and pre-flatten → architecture-after. The intermediate optimization has
its own parent comparison.

## Compare revisions

- [Integrate full Prefill and Decode graphs](../../tasks/deepseek-v4-1-flash-a3-architecture-integration/README.md) · c43f180 → df1307d
- [Flatten the DeepSeek A3 model graph](../../tasks/deepseek-v4-1-flash-a3-architecture-flattening/README.md) · 4dfe05f → 3acf570
- [Prefill memory movement](../../tasks/deepseek-v4-1-flash-a3-prefill-memory-movement/README.md) · d6caf0f → fa5dfb6
- [Prefill GEMM projections](../../tasks/deepseek-v4-1-flash-a3-prefill-gemm/README.md) · fa5dfb6 → 120f3a0
- [Prefill vector kernels](../../tasks/deepseek-v4-1-flash-a3-prefill-vector/README.md) · 120f3a0 → 0154cd4
- [Prefill Engram unrotation](../../tasks/deepseek-v4-1-flash-a3-prefill-engram/README.md) · 7d6e410 → da9b4ab
- [Prefill MoE routing panels](../../tasks/deepseek-v4-1-flash-a3-prefill-moe-panels/README.md) · 7dea2f4 → 7d6e410
- [Prefill expert GEMM](../../tasks/deepseek-v4-1-flash-a3-prefill-expert-gemm/README.md) · 22281c2 → 2ae6005
- [Prefill Indexer scheduling](../../tasks/deepseek-v4-1-flash-a3-prefill-indexer/README.md) · c74d9bc → 9c05078
- [Prefill Indexer scoring and Query FP4](../../tasks/deepseek-v4-1-flash-a3-prefill-indexer-fp4/README.md) · d572e5b → bdd156b
- [Prefill mHC kernels](../../tasks/deepseek-v4-1-flash-a3-prefill-mhc/README.md) · da9b4ab → 22281c2
- [Prefill mHC Sinkhorn reductions](../../tasks/deepseek-v4-1-flash-a3-prefill-mhc-sinkhorn/README.md) · 78503a9 → 1b00b94
- [Prefill sparse attention](../../tasks/deepseek-v4-1-flash-a3-prefill-sparse-attention/README.md) · e48daef → 0b8c1a6
- [SWA window attention on Cube](../../tasks/deepseek-v4-1-flash-a3-swa-cube/README.md) · 30ff61c → 78503a9
- [Decode projection K tiles](../../tasks/deepseek-v4-1-flash-a3-decode-projection-k/README.md) · 4d0dbbc → e48daef
- [Fuse Decode KV normalization and RoPE](../../tasks/deepseek-v4-1-flash-a3-decode-kv-norm-rope/README.md) · 79fd9d8 → 4157802
- [Decode RoPE stores](../../tasks/deepseek-v4-1-flash-a3-decode-rope-stores/README.md) · 59bf9b8 → 7dea2f4
- [Separate Decode mHC kernels](../../tasks/deepseek-v4-1-flash-a3-decode-mhc-split/README.md) · 9c05078 → d572e5b
- [Decode mHC Post tiling](../../tasks/deepseek-v4-1-flash-a3-decode-mhc-post/README.md) · fd4620b → a6a3ba7
- [Prefill MoE routing and dispatch](../../tasks/deepseek-v4-1-flash-a3-prefill-moe-routing/README.md) · a6a3ba7 → 2bd279d

## Build and report

Use the Playground CLI with this repository as the task root. The modules
are pinned to published commits. Use a GitHub account with read access to
the artifact. [Infra Playground](https://github.com/co-agent-serving/infra-playground)

```bash
python3 "$PLAYGROUND/playground" --root "$ARTIFACT" build \
  --task deepseek-v4-1-flash-a3-prefill-memory-movement
python3 "$PLAYGROUND/playground" report --catalog "$ARTIFACT" \
  --output "$ARTIFACT/report.html"
```

The task files describe both revisions, the workload and the acceptance
checks. Measured results enter the report after export with evidence.

## Evidence format

- evidence/fixtures.json contains a cases object keyed by the IDs in
  input.json. Record each production entry, shapes, layout, valid rows,
  initial values, boundary inputs and numerical reference.
- evidence/baseline.json and evidence/candidate.json each contain revision,
  fixture_sha256 (the hash of the exact fixture file), devices, toolchain,
  checkpoint_identity and cases. Each case contains correctness_passed,
  correctness_reference and latency_us (three lists of raw samples).
- evidence/comparison.json is written by the pinned tools/compare.py. It
  validates the pair and reports each median and latency ratio. A ratio
  below one indicates an improvement.

Export the validated comparison with the Playground workflow. Use matching
fixtures and measurement boundaries for any vLLM-Ascend reference.
