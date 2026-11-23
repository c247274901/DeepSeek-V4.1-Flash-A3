# DeepSeek V4.1 Flash A3 PyPTO Serving

Standalone W8A8 text generation for sixteen Ascend A3 devices, with DP4, TP4
and EP16. The artifact contains the model kernels, offline generation,
HTTP serving and workload definitions for comparing the retained revisions.

## Environment

Use Python 3.10 or later, Torch, CANN and the PyPTO revision recorded in
[pypto.pin](pypto-serving-deepseek-v4.1-flash-a3/pypto.pin). Install that
revision with its pinned Simpler runtime and PTOAS. The runtime and its
toolchain are available from [PyPTO](https://github.com/hw-native-sys/pypto).
Install the service's Python dependencies in the same environment.

```bash
python -m pip install -r pypto-serving-deepseek-v4.1-flash-a3/requirements.txt
```

Pass the external DeepSeek-V4.1-Flash W8A8 checkpoint and a sixteen-device
allocation. The service uses the checkpoint tokenizer and text renderer.
Engram reads selected rows from the checkpoint on the host. Checkpoint
weights and compiler outputs are external to this repository.

## Start the service

```bash
bash pypto-serving-deepseek-v4.1-flash-a3/run.sh \
  --model-root "$CHECKPOINT" --devices "$TASK_DEVICE" \
  --dp 4 --tp 4 --ep 16 --max-model-len 8200 --chunk-size 8192 \
  --prefill-moe-slab-rows 128 --host 127.0.0.1 --port 8000
```

The launcher compiles the two graphs into a fresh directory under
`PYPTO_SERVING_CACHE_DIR` or the user cache. To reuse compiled outputs, pass
`--graph-dir` with the previously printed path and the same context, chunk
capacity, MoE slab, environment and device topology. It checks both native
manifests before loading weights. `/health` reports ready after the worker
loads the checkpoint and prepares communication windows.

```bash
curl -sS http://127.0.0.1:8000/health
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-A3","messages":[{"role":"user","content":"Explain tensor parallelism briefly."}],"temperature":0,"max_tokens":32,"stream":true,"stream_options":{"include_usage":true}}'
```

`/v1/completions` accepts text or a single list of token IDs. Both endpoints
support greedy generation, EOS stopping, `ignore_eos` and optional returned
token IDs. `/metrics` reports queue counts and runtime configuration. The
native graphs support one active request and a context of up to 16384
positions including output, with chunk size up to 8192. The server queues up to eight
waiting requests by default and clears the request's KV and Engram history
before reusing the slot. Continuous batching and speculation are outside
this adapter's supported interface.

For offline replay and compiled graph commands, see the
[model runbook](pypto-serving-deepseek-v4.1-flash-a3/models/deepseek_v4_1_flash_a3/README.md).

## History and validation

The kernel snapshot is tagged `release/1122-kernels` and comes from
`Adelaide999/pypto-lib` at
`38c77a9817090609d74728f24bccea9fb1a682ac`. The serving adapter reuses the
DeepSeek chat renderer and request semantics from `Adelaide999/pypto-serving`
at `cd818945c3c4e5454b8c259306d4e6f9e865e05c` and binds them to the 1122
resident programs. The model history preserves individual optimization,
correctness and revert commits with original author and source SHA.

Architecture baselines, operator comparisons, and export instructions for
Blueprint and the Playground web report are in the [task catalog](tasks/README.md).

The 8192-row Prefill graph (MoE slab 128) and the 16-row Decode graph
compiled for A3 at context capacity 8200. Host tests cover the serving routes
and native Decode staging. Numerical acceptance and
serving latency on sixteen devices require the checkpoint and a free allocation.
