"""Decode benchmark connector with an explicit external-history length."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorScheduler,
)

from .config import validate_history_tokens

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


class HistoryDecodeBenchConnectorScheduler(DecodeBenchConnectorScheduler):
    """Use request ``history_tokens`` instead of filling the entire prefix."""

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if request.request_id in self._filled_requests:
            return 0, False
        params = request.kv_transfer_params or {}
        if "history_tokens" not in params:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        history_tokens = validate_history_tokens(
            params["history_tokens"], request.num_tokens
        )
        remaining_history = max(0, history_tokens - num_computed_tokens)
        return remaining_history, False


class HistoryDecodeBenchConnector(DecodeBenchConnector):
    """Fill only the history prefix explicitly declared by each request."""

    def __init__(self, vllm_config: "VllmConfig", role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_scheduler is not None:
            self.connector_scheduler = HistoryDecodeBenchConnectorScheduler(
                vllm_config
            )
