"""vLLM plugin registration for the in-repository DCP benchmark."""

from __future__ import annotations

import os


BACKEND_CLASS_PATHS = {
    "mega": "min_fa3_vllm_plugin.mega_backend.MegaDCPAttentionBackend",
    "mega-fa3-native": (
        "min_fa3_vllm_plugin.mega_backend.MegaDCPAttentionBackend"
    ),
    "vllm-ag-rs": (
        "min_fa3_vllm_plugin.mega_backend.VLLMAGRSDCPAttentionBackend"
    ),
    "vllm-a2a": "min_fa3_vllm_plugin.mega_backend.VLLMA2ADCPAttentionBackend",
}


def selected_backend_class_path() -> str:
    backend = os.environ.get("MIN_FA3_DCP_BACKEND", "mega")
    try:
        return BACKEND_CLASS_PATHS[backend]
    except KeyError as exc:
        choices = ", ".join(BACKEND_CLASS_PATHS)
        raise ValueError(
            f"MIN_FA3_DCP_BACKEND must be one of {choices}, got {backend!r}"
        ) from exc


def register() -> None:
    """Register the selected attention backend and benchmark KV connector."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from .moe_compat import install_moe_compat

    install_moe_compat()
    if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY") == "min_fa3_balanced":
        from .balanced_routing import register_balanced_routing

        register_balanced_routing()

    register_backend(
        AttentionBackendEnum.CUSTOM,
        selected_backend_class_path(),
    )
    connector_name = "HistoryDecodeBenchConnector"
    if connector_name not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            connector_name,
            "min_fa3_vllm_plugin.history_connector",
            connector_name,
        )


__all__ = ["BACKEND_CLASS_PATHS", "register", "selected_backend_class_path"]
