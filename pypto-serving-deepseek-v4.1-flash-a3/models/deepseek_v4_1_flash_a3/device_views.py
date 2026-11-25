# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed views into shared resident device allocations."""

from pypto.runtime import DeviceTensor, StackedDeviceTensor


def _device_view(allocation, value, base_offset=0):
    """Bind a contiguous typed view to the allocation's owning Buffer."""
    offset = value.storage_offset() * value.element_size() - base_offset
    if offset < 0:
        raise ValueError("Resident tensor view begins before its allocation")
    if offset + value.numel() * value.element_size() > allocation.nbytes:
        raise ValueError("Resident tensor view exceeds its allocation")
    return DeviceTensor(allocation.data_ptr + offset, value.shape, value.dtype,
                        buffer=allocation.buffer, byte_offset=allocation.byte_offset + offset)


class _EngramRankShards(StackedDeviceTensor):
    """Bind logical dimensions for validation and NZ blocks for dispatch."""

    def __getitem__(self, index):
        leading = index[0] if isinstance(index, tuple) and index else index
        if isinstance(leading, slice):
            rank = leading.start
            if type(rank) is int and leading == slice(rank, rank + 1):
                trailing = index[1:] if isinstance(index, tuple) else ()
                shard = super().__getitem__(rank)
                rows, columns = shard.shape
                blocks = (columns // 16, rows // 16, 16, 16)
                if len(trailing) > len(blocks) or any(
                    part not in (slice(None), slice(0, dim))
                    for part, dim in zip(trailing, blocks)
                ):
                    raise ValueError("The Engram NZ view requires a whole-shard slice")
                return DeviceTensor(shard.data_ptr, (1, *blocks), shard.dtype,
                                    buffer=shard.buffer, byte_offset=shard.byte_offset)
        return super().__getitem__(index)


def stack_rank_tensor(name, shards):
    """Bind the full argument shape to its rank-local resident allocations."""
    tensor_type = _EngramRankShards if name == "engram_wkv" else StackedDeviceTensor
    return tensor_type(shards, (len(shards), *shards[0].shape), tuple(range(len(shards))))
