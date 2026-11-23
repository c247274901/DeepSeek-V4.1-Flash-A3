# Reproduce the comparison

Verify both checkout SHAs against [input.json](input.json). Inspect the
published candidate diff and its original commit message.

```bash
git -C modules/artifact-candidate show 22281c2804663f3760f99a320a0988b1294a6c80
```

Start with 2048 live rows per TP4 rank and 8192 after a global gather where applicable.
Freeze the observed shapes, layout, valid counts, offsets and initial buffers.
Replay the production entry on one A3 device when it is rank-local, or with
the same allocated topology for a collective. Check each output against
its native Torch golden or an independent reference.

Use three paired timing windows with five warmups and 100
measured calls per window. Keep compilation, checkpoint load and profiling
outside the timing. Store the raw samples and validate the paired runs using
the [evidence format](../README.md).

```bash
python3 modules/artifact-harness/tools/compare.py \
  --input .agents/task/input.json --evidence evidence
```

The candidate changes these model files.

- decode_c1a_full.py
- decode_c1a_reindex.py
- decode_c1a_reuse.py
- decode_c2a_full.py
- decode_c2a_reuse.py
- decode_fwd.py
- decode_swa.py
- hc.py
- moe.py
- prefill_c1a_full.py
- prefill_c1a_reindex.py
- prefill_c1a_reuse.py
- prefill_c2a_full.py
- prefill_c2a_reuse.py
- prefill_fwd.py
- prefill_swa.py
