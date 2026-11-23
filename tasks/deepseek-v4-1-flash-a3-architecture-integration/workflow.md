# Reproduce the comparison

Verify both checkout SHAs against [input.json](input.json). Inspect the
published candidate diff and its original commit message.

```bash
git -C modules/artifact-candidate show df1307df08fb694df5a2a8f71d66c67eee708f17
```

Record the full-length prompt and a fixed Decode trace. Use each revision's
model runbook and separate compile directories. Measure the complete Prefill
phase and seven Decode steps on the same 16-device allocation with resident
weights. Check logits with the same independent checkpoint reference.

Use three paired timing windows with five warmups and 30
measured calls per window. Keep compilation, checkpoint load and profiling
outside the timing. Store the raw samples and validate the paired runs using
the [evidence format](../README.md).

```bash
python3 modules/artifact-harness/tools/compare.py \
  --input .agents/task/input.json --evidence evidence
```

The candidate changes these model files.

- MODULE_ALIGNMENT.md
- README.md
- __init__.py
- bounded_decode.py
- cache.py
- decode_fwd.py
- decode_replay_inputs.py
- driver.py
- generate.py
- generation.py
- host.py
- network.py
- operators/__init__.py
- operators/communication.py
- operators/compressor.py
- operators/config.py
- operators/decode_c1a_full.py
- operators/decode_c1a_reindex.py
- operators/decode_c1a_reuse.py
- operators/decode_c2a_full.py
- operators/decode_c2a_reuse.py
- operators/decode_indexer.py
- operators/decode_swa.py
- operators/engram.py
- operators/expert_routed.py
- operators/expert_shared.py
- operators/gate.py
- operators/gemm.py
- operators/hc.py
- operators/indexer_reference.py
- operators/input_pack.py
- operators/lm_head.py
- operators/moe.py
- operators/o_proj.py
- operators/prefill_c1a_full.py
- operators/prefill_c1a_reindex.py
- operators/prefill_c1a_reuse.py
- operators/prefill_c2a_full.py
- operators/prefill_c2a_reuse.py
- operators/prefill_cp_allgather.py
- operators/prefill_indexer.py
- operators/prefill_swa.py
- operators/qkv_proj_rope.py
- operators/quant.py
- operators/rmsnorm.py
- operators/rope_tables.py
- operators/softmax_precision.py
- operators/sparse_attention.py
- operators/utils.py
- operators/weights.py
- prefill_compressed_runtime.py
- prefill_consumer_runtime.py
- prefill_fwd.py
- prefill_layer.py
- prefill_moe.py
- prefill_resident.py
- reference.py
- replay.py
- resident_decode.py
- resident_decode_entry.cpp
- resident_memory.py
- resident_mesh.py
- routed_exchange.py
- staged_generate.py
- staged_prefill.py
- state.py
- swimlane.py
- telemetry.py
- tp_exchange.py
- validation.py
- validation_data/input_ids.json
- weights.py
