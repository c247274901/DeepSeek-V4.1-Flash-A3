# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind a long-lived service to the native 1122 Prefill and Decode programs."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from tokenizers import Tokenizer

from .encoding import encode_messages


@dataclass(frozen=True)
class LaunchConfig:
    model_root: Path
    devices: tuple[int, ...]
    context: int
    chunk_size: int
    slab_rows: int
    graph_dir: Path | None
    cache_dir: Path


class NativeSession:
    """Own one request slot and reuse the same device resources across requests."""

    def __init__(self, graphs, tokenizer, eos_id, config, metadata):
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.config = config
        self.metadata = metadata
        self.memory = SimpleNamespace(phase="ready")
        self.needs_reset = False

    def encode(self, prompt, *, chat=False):
        if chat:
            return self.tokenizer.encode(encode_messages(prompt), add_special_tokens=False).ids
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt).ids
        return prompt

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(self, request_id, tokens, max_tokens, ignore_eos, on_token, cancelled):
        from concurrent.futures import CancelledError

        if cancelled():
            raise CancelledError()
        if self.needs_reset:
            self.graphs.reset()
        self.needs_reset = True
        request = {"request_id": request_id, "dp_rank": 0, "input_ids": tokens}
        options = SimpleNamespace(
            capacity=self.metadata["prefill_capacity"], decode_capacity=16,
            chunk_size=self.config.chunk_size, context_capacity=self.config.context,
            max_new_tokens=max_tokens, ignore_eos=ignore_eos,
        )
        return self.graphs.replay(
            self.tokenizer, request, self.memory, {}, options=options, label=request_id,
            on_token=on_token, cancelled=cancelled, collect_routing=False,
        )


@contextmanager
def open_session(settings):
    """Compile or validate artifacts, then bind the checkpoint once."""
    from models.deepseek_v4_1_flash_a3 import config as model
    from models.deepseek_v4_1_flash_a3.driver import compile_generation_graphs, load_manifest

    prefill_capacity = ((settings.chunk_size + 15) // 16) * 16
    graph_dir = settings.graph_dir
    if graph_dir is None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        graph_dir = Path(tempfile.mkdtemp(prefix="graphs-", dir=settings.cache_dir))
        print(f"A3_SERVING_GRAPH_DIR {graph_dir}", flush=True)
        compile_generation_graphs(
            graph_dir / "prefill", graph_dir / "decode", devices=",".join(map(str, settings.devices)),
            context=settings.context, prefill_capacity=prefill_capacity, prefill_slab_rows=settings.slab_rows,
        )
    prefill_dir, decode_dir = graph_dir / "prefill", graph_dir / "decode"
    prefill = load_manifest(prefill_dir, mode="prefill", capacity=prefill_capacity)
    decode = load_manifest(decode_dir, mode="decode", capacity=16)
    if prefill["history_capacity"] != settings.context or decode["history_capacity"] != settings.context:
        raise ValueError("Both graphs must match --max-model-len")
    if prefill["moe_slab_rows"] != settings.slab_rows:
        raise ValueError("Prefill graph must match --prefill-moe-slab-rows")

    model.set_moe_tokens(settings.slab_rows)
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.generation import open_resident_graphs
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

    checkpoint = Checkpoint(settings.model_root)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer_config = json.loads((checkpoint.root / "tokenizer_config.json").read_text())
    eos = tokenizer_config["eos_token"]
    eos_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
    if eos_id is None:
        raise ValueError("Checkpoint EOS token is missing from the tokenizer")
    options = SimpleNamespace(
        runtime_dir=prefill_dir, decode_runtime_dir=decode_dir, profile_all_steps=None,
        capacity=prefill_capacity, decode_capacity=16, context_capacity=settings.context, max_new_tokens=256,
    )
    metadata = {
        "checkpoint_identity": checkpoint.identity,
        "graph_dir": str(graph_dir),
        "context_capacity": settings.context,
        "prefill_capacity": prefill_capacity,
        "prefill_chunk_size": settings.chunk_size,
        "prefill_moe_slab_rows": settings.slab_rows,
        "vocab_size": model.FLASH.vocab_size,
        "topology": prefill["topology"],
    }
    weights = ResidentWeights(checkpoint)
    engram = HostEngram(checkpoint, tokenizer)
    memory = SimpleNamespace(phase="prepare", sample=lambda: None)
    with open_resident_graphs(options, list(settings.devices), weights, engram, memory, metadata) as graphs:
        yield NativeSession(graphs, tokenizer, eos_id, settings, metadata)
