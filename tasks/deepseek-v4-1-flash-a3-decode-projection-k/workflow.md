# Reproduce the comparison

Verify both checkout SHAs against [input.json](input.json). Inspect the
published candidate diff and its original commit message.

```bash
git -C modules/artifact-candidate show e48daefd4df711b1232719eaaa705490179fba07
```

Start with one live Decode row in a 16-row allocation.
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

- qkv_proj_rope.py
