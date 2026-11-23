# Reproduce the comparison

Verify both checkout SHAs against [input.json](input.json). Inspect the
published candidate diff and its original commit message.

```bash
git -C modules/artifact-candidate show 3acf570d28f2bca3a7821c598e136088b8f8e050
```

Record the full-length prompt and a fixed Decode trace. Use each revision's
model runbook and separate compile directories. Measure the complete Prefill
phase and seven Decode steps on the same 16-device allocation with resident
weights. Check logits with the same independent checkpoint reference.

Use three paired timing windows with five warmups and 30
measured calls per window. Keep compilation, checkpoint load and profiling
outside the timing. Store the raw samples and validate the paired runs using
the [evidence format](../../docs/blueprint/README.md).

```bash
python3 modules/artifact-harness/tools/compare.py \
  --input .agents/task/input.json --evidence evidence
```

The candidate changes these model files.

- MODULE_ALIGNMENT.md
- README.md
- __init__.py
- activation.py
- activation_quant.py
- blocked_rmsnorm.py
- bounded_decode.py
- cache.py
- cache_pool.py
- compressor.py
- config.py
- decode_c1a.py
- decode_c1a_full.py
- decode_c1a_reindex.py
- decode_c1a_reuse.py
- decode_c2a.py
- decode_c2a_full.py
- decode_c2a_reuse.py
- decode_compressed_attention.py
- decode_config.py
- decode_fwd.py
- decode_indexer.py
- decode_layer.py
- decode_layer_config.py
- decode_moe.py
- decode_moe_ep.py
- decode_network.py
- decode_replay_inputs.py
- decode_sparse_attention.py
- decode_swa.py
- device_views.py
- driver.py
- engram.py
- engram_host.py
- expert_routed.py
- expert_shared.py
- gate.py
- gemm.py
- generate.py
- generation.py
- grouped_moe.py
- hc.py
- host.py
- indexer.py
- indexer_reference.py
- input_ids.json
- input_pack.py
- kernels/prefill_output_projection.cpp
- lm_head.py
- metadata.py
- moe.py
- moe_transport.py
- network.py
- o_proj.py
- operator_config.py
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
- paged_indexer.py
- prefill.example.json
- prefill_c1a_full.py
- prefill_c1a_reindex.py
- prefill_c1a_reuse.py
- prefill_c2a.py
- prefill_c2a_full.py
- prefill_c2a_reuse.py
- prefill_compressed_reuse.py
- prefill_compressed_runtime.py
- prefill_consumer_runtime.py
- prefill_cp_allgather.py
- prefill_fwd.py
- prefill_indexer.py
- prefill_indexer_validation.py
- prefill_layer.py
- prefill_layer_plan.json
- prefill_moe.py
- prefill_network_engram.py
- prefill_prepare.py
- prefill_resident.py
- prefill_sparse_attention.py
- prefill_swa.py
- prefill_vector_launch.cpp
- prefill_vector_launch.py
- qkv_proj_rope.py
- quant.py
- reference.py
- replay.py
- resident_decode.py
- resident_decode_entry.cpp
- resident_memory.py
- resident_mesh.py
- resident_weights.py
- rmsnorm.py
- rope.py
- rope_tables.py
- routed_exchange.py
- runtime_config.py
- softmax_precision.py
- sparse_attention.py
- staged_generate.py
- staged_prefill.py
- state.py
- swa.py
- tp_exchange.py
- utils.py
- validation.py
- validation_data/input_ids.json
- weights.py
