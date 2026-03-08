# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, NamedTuple, Optional

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import load_cache_to_device_buffer_mla
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: HiSparseTokenToKVPoolAllocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group: torch.distributed.ProcessGroup,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device

        self.mem_pool_device: HiSparseNSATokenToKVPool = (
            self.token_to_kv_pool_allocator.get_kvcache()
        )
        self.mem_pool_host = MLATokenToKVPoolHost(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=2,
            host_size=0,
            page_size=1,  # for simplicity, we set page size to 1 to enable backup one token at a time
            layout="layer_first",
            override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
        )

        max_num_reqs = req_to_token_pool.size
        max_context_len = req_to_token_pool.max_context_len

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size), dtype=torch.int64, device=device
        )
        self.req_to_host_pool = torch.full(
            (max_num_reqs, max_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (max_num_reqs, layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (max_num_reqs, layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.bitmap = torch.full(
            (max_num_reqs, max_context_len),
            -1,
            dtype=torch.int16,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(max_num_reqs, layer_num, 1)
            .contiguous()
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_reqs, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def admit_request_into_staging(self, req: Req) -> None:
        req.staging = True
        logical_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(
            logical_indices
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len)
        if host_indices is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d tokens (req %s)",
                prefill_len,
                req.rid,
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {prefill_len} tokens"
            )
        host_indices = host_indices.to(device=self.device)
        self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, io_backend="kernel"
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def alloc_device_buffer(self, req: Req) -> None:
        allocated_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ]
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            allocated_indices,
            self.padded_buffer_size,
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(kv_allocated_len=%d, padded_buffer_size=%d)",
                req.rid,
                req.kv_allocated_len,
                self.padded_buffer_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")
        self.req_to_device_buffer[req.req_pool_idx, : self.padded_buffer_size] = (
            buffer_indices
        )

        # initialize the token locs for the device buffer
        self.req_device_buffer_tokens[
            req.req_pool_idx, :, : self.device_buffer_size
        ] = torch.arange(self.device_buffer_size, device=self.device)
        self.req_device_buffer_token_locs[
            req.req_pool_idx, :, : self.padded_buffer_size
        ] = buffer_indices[: self.padded_buffer_size]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_batch(self) -> Optional[ScheduleBatch]:
        ready_batch = None
        if len(self.ack_staging_queue) == 0:
            return ready_batch

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            req.staging = False
            finish_count -= 1
            if (
                len(self.ack_staging_queue) == 0
                or self.ack_staging_queue[0][2].batch != req.batch
            ):
                if ready_batch is None:
                    ready_batch = req.batch
                else:
                    ready_batch.merge_batch(req.batch)
            # to break the circular reference
            req.batch = None
        return ready_batch

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(seq_lens, req_pool_indices)

        reserved_buffer_loc = self.req_to_device_buffer[
            req_pool_indices, self.device_buffer_size
        ]

        short_reqs = seq_lens <= self.device_buffer_size
        if torch.any(short_reqs):
            reserved_buffer_loc[short_reqs] = self.req_to_device_buffer[
                req_pool_indices[short_reqs], seq_lens[short_reqs] - 1
            ]

        # todo, clear the prior mapping as well
        self.mem_pool_device.full_to_hisparse_device_index_mapping[out_cache_loc] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        """Back up the previous decode token from the reserved device buffer slot.

        Called at the start of map_last_loc_to_buffer (inside prepare_for_decode)
        so the backup runs on the default stream *before* the forward overwrites
        the reserved slot.

        For long sequences, at every decode step the token at position
        seq_len - 2 needs a host backup — except on the very first decode,
        where that position is a prefill token already backed up during
        staging (detected by req_to_host_pool >= 0).
        """
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)

        prev_pos = seq_lens - 2
        long_mask = (prev_pos >= 0) & (seq_lens > self.device_buffer_size)
        if not torch.any(long_mask):
            return

        candidate_reqs = req_pool_indices[long_mask]
        candidate_pos = prev_pos[long_mask]

        # Skip positions already backed up (first decode after prefill)
        needs_backup = self.req_to_host_pool[candidate_reqs, candidate_pos] < 0
        if not torch.any(needs_backup):
            return

        backup_req_indices = candidate_reqs[needs_backup]
        backup_positions = candidate_pos[needs_backup]

        device_locs = self.req_to_device_buffer[
            backup_req_indices, self.device_buffer_size
        ]

        host_locs = self.mem_pool_host.alloc(len(device_locs))
        if host_locs is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d decode backup tokens",
                len(device_locs),
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {len(device_locs)} decode backup tokens"
            )
        host_locs = host_locs.to(device=self.device)
        self.req_to_host_pool[backup_req_indices, backup_positions] = host_locs

        # Runs on the default stream; forward_stream.wait_stream(default_stream)
        # ensures the backup completes before the forward overwrites the slot.
        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_locs,
            device_locs.contiguous(),
            io_backend="kernel",
        )

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        # a dummy selection for testing
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )
        for i in range(num_reqs):
            top_n = min(
                seq_lens[i],
                self.top_k,
            )
            if top_n == 0:
                continue
            top_k_indices[i, :top_n] = self.req_to_device_buffer[req_pool_indices[i]][
                :top_n
            ]
        return top_k_indices

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.
        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            # Validate token positions
            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                # Short sequence: all tokens reside in the device buffer,
                # directly indexed by token position.
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                # Long sequence: latest token uses reserved slot; all others
                # are loaded from host into reusable device buffer slots.
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                # Assign the reserved buffer slot for the latest token
                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                # Load all other tokens from host
                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    # Reuse the first num_to_load slots in the device buffer
                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def retract_req(self, req: Req) -> None:
        self.request_finished(req)

    def request_finished(self, req: Req):
        # release memory
        buffer_indices = self.req_to_device_buffer[req.req_pool_idx]
        self.token_to_kv_pool_allocator.free_hisparse_indices(buffer_indices)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ]
        self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            allocated_locs
        ] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        # clear req info
        self.req_device_buffer_tokens[req.req_pool_idx, :, :] = -1
        self.req_device_buffer_token_locs[req.req_pool_idx, :, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.lru_slots[req.req_pool_idx].copy_(self._lru_init)

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        num_reqs = req_pool_indices.size(0)
        # Reuse pre-allocated buffer (CUDA-graph safe: stable address)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)
        block_size = 512
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens,
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs,
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            diff_map=self.bitmap,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=self.lru_slots,
            layer_id=layer_id,
            item_size_bytes=self.mem_pool_host.token_stride_size,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices
