"""T1 controls adapted from ring_test/allgather_attention.py; evaluation is unchanged."""
import torch
from ring_test.allgather_attention import AllGatherAttention as BaseAllGather


class AllGatherControl(BaseAllGather):
    def prepare_compute(self):
        self.preloaded = {}
        for head in range(0, self.k.size(1), self.heads_k_stride):
            work = self._start_kv_all_gather(0, head)
            self._wait_kv_all_gather(work)
            self._order_kv_chunk(0)
            self.preloaded[head] = self.kv_ordered[0].clone()

    def forward(self, *, overlap: bool = True, compute_only: bool = False) -> torch.Tensor:
        self._forward_ready = False
        current_buffer = 0
        current_work = self._start_kv_all_gather(current_buffer, 0) if not compute_only else None
        for kv_head_start in range(0, self.k.size(1), self.heads_k_stride):
            if compute_only:
                self.kv_ordered[current_buffer].copy_(self.preloaded[kv_head_start])
            else:
                self._wait_kv_all_gather(current_work)
                self._order_kv_chunk(current_buffer)

            q_head_slice = self._q_head_slice(kv_head_start)
            self.q_chunk.copy_(self.q[:, q_head_slice])
            next_kv_head_start = kv_head_start + self.heads_k_stride
            next_buffer = 1 - current_buffer
            if not compute_only and overlap and next_kv_head_start < self.k.size(1):
                current_work = self._start_kv_all_gather(
                    next_buffer, next_kv_head_start
                )

            if not self.causal:
                out, lse = self._run_forward_block(
                    self.q_chunk,
                    self.kv_ordered[current_buffer, 0],
                    self.kv_ordered[current_buffer, 1],
                    self.full_cu_q,
                    self.global_cu_k,
                    self.full_cu_q_host,
                    self.global_cu_k_host,
                    self.local_seqlen,
                    self.global_seqlen,
                    False,
                )
                self.out[:, q_head_slice].copy_(out)
                self.forward_lse_front[q_head_slice].copy_(lse)
                if not compute_only and not overlap and next_kv_head_start < self.k.size(1):
                    current_work = self._start_kv_all_gather(
                        next_buffer, next_kv_head_start
                    )
                current_buffer = next_buffer
                continue

            ordered_k = self.kv_ordered[current_buffer, 0].view(
                self.batch_size,
                self.global_seqlen,
                self.heads_k_stride,
                self.k.size(2),
            )
            ordered_v = self.kv_ordered[current_buffer, 1].view_as(ordered_k)
            q = self.q_chunk.view(
                self.batch_size, self.local_seqlen, self.q_heads_per_chunk, self.q.size(2)
            )
            q_front = self.q_front.view(
                self.batch_size, self.half, self.q_heads_per_chunk, self.q.size(2)
            )
            q_back = self.q_back.view_as(q_front)
            k_front = self.k_front.view(
                self.batch_size, -1, self.heads_k_stride, self.k.size(2)
            )
            v_front = self.v_front.view_as(k_front)
            k_back = self.k_back.view(
                self.batch_size, -1, self.heads_k_stride, self.k.size(2)
            )
            v_back = self.v_back.view_as(k_back)
            for batch_idx in range(self.batch_size):
                q_front[batch_idx].copy_(q[batch_idx, : self.half])
                q_back[batch_idx].copy_(q[batch_idx, self.half :])
                k_front[batch_idx].copy_(ordered_k[batch_idx, : k_front.size(1)])
                v_front[batch_idx].copy_(ordered_v[batch_idx, : v_front.size(1)])
                k_back[batch_idx].copy_(ordered_k[batch_idx, : k_back.size(1)])
                v_back[batch_idx].copy_(ordered_v[batch_idx, : v_back.size(1)])

            out_front, lse_front = self._run_forward_block(
                self.q_front,
                self.k_front,
                self.v_front,
                self.half_cu_q,
                self.front_cu_k,
                self.half_cu_q_host,
                self.front_cu_k_host,
                self.half,
                self.k_front.size(0) // self.batch_size,
                True,
            )
            out_back, lse_back = self._run_forward_block(
                self.q_back,
                self.k_back,
                self.v_back,
                self.half_cu_q,
                self.back_cu_k,
                self.half_cu_q_host,
                self.back_cu_k_host,
                self.half,
                self.k_back.size(0) // self.batch_size,
                True,
            )
            out_view = self.out.view(
                self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2)
            )
            packed_front = out_front.view(
                self.batch_size, self.half, self.q_heads_per_chunk, self.q.size(2)
            )
            packed_back = out_back.view_as(packed_front)
            for batch_idx in range(self.batch_size):
                out_view[batch_idx, : self.half, q_head_slice].copy_(
                    packed_front[batch_idx]
                )
                out_view[batch_idx, self.half :, q_head_slice].copy_(
                    packed_back[batch_idx]
                )
            self.forward_lse_front[q_head_slice].copy_(lse_front)
            assert self.forward_lse_back is not None
            self.forward_lse_back[q_head_slice].copy_(lse_back)
            if not compute_only and not overlap and next_kv_head_start < self.k.size(1):
                current_work = self._start_kv_all_gather(
                    next_buffer, next_kv_head_start
                )
            current_buffer = next_buffer

        self._forward_ready = True
        return self.out

