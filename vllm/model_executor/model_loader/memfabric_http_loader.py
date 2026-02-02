# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import socket
import threading
import time
import urllib.request
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = init_logger(__name__)


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: int = 5) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp_data = resp.read().decode("utf-8")
    if not resp_data:
        return {}
    return json.loads(resp_data)


def _resolve_node_ip(extra: dict[str, Any]) -> str:
    if extra.get("node_ip"):
        return str(extra["node_ip"])
    for key in ("VLLM_NODE_IP", "HOST_IP", "POD_IP"):
        if key in os.environ:
            return os.environ[key]
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _get_rank_info() -> dict[str, int]:
    rank = 0
    local_rank = 0
    tp_rank = 0
    pp_rank = 0
    dp_rank = 0
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        pass
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group

        tp_rank = get_tensor_model_parallel_rank()
        local_rank = get_tp_group().local_rank
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import get_pp_group

        pp_rank = get_pp_group().rank_in_group
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import get_dp_group

        dp_rank = get_dp_group().rank_in_group
    except Exception:
        pass
    return {
        "rank": rank,
        "local_rank": local_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": dp_rank,
    }


def _build_model_key(vllm_config: VllmConfig, model_config: ModelConfig) -> dict[str, Any]:
    parallel = vllm_config.parallel_config
    key = {
        "model": model_config.model,
        "revision": model_config.revision,
        "dtype": str(model_config.dtype),
        "quant": model_config.quantization,
        "tp": parallel.tensor_parallel_size,
        "pp": parallel.pipeline_parallel_size,
        "pcp": parallel.prefill_context_parallel_size,
        "model_impl": model_config.model_impl,
    }
    arch = getattr(model_config.hf_config, "architectures", None)
    if arch:
        key["architectures"] = arch
    return key


def _get_npu_id(extra: dict[str, Any]) -> int:
    if "npu_id" in extra:
        return int(extra["npu_id"])
    try:
        return int(torch.npu.current_device())
    except Exception:
        return 0


def _build_my_id(extra: dict[str, Any], node_ip: str, rank: int) -> str:
    if extra.get("my_id"):
        return str(extra["my_id"])
    base_port = int(extra.get("base_port", 10000))
    return f"{node_ip}:{base_port + rank}"


def _params_metadata(model: nn.Module) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        if p.numel() == 0:
            continue
        params.append(
            {
                "name": name,
                "dtype": str(p.dtype),
                "shape": list(p.shape),
                "numel": int(p.numel()),
                "bytes": int(p.numel() * p.element_size()),
                "addr": int(p.data_ptr()),
                "device": str(p.device),
            }
        )
    return params


def _register_memory(engine: Any, params: list[dict[str, Any]]):
    for p in params:
        engine.register_memory(int(p["addr"]), int(p["bytes"]))


def _initialize_engine(extra: dict[str, Any], my_id: str, npu_id: int, role: str):
    try:
        from memfabric_hybrid import (
            TransferEngine,
            create_config_store,
            set_conf_store_tls,
            set_log_level,
        )
    except Exception as e:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(f"memfabric_hybrid import failed: {e}") from e

    store_url = extra.get("store_url")
    if not store_url:
        raise ValueError("model_loader_extra_config.store_url is required")
    set_log_level(int(extra.get("log_level", 1)))
    set_conf_store_tls(False, "")
    create_config_store(store_url)
    time.sleep(1)

    engine = TransferEngine()
    op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
    ret = engine.initialize(store_url, my_id, role, int(npu_id), op_type)
    if ret != 0:
        raise RuntimeError(f"TransferEngine initialize failed ret={ret}")
    return engine


class MemfabricHttpLoader(BaseModelLoader):
    """Load weights by coordinating memfabric D2D transfers via an HTTP control plane."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._default_loader = DefaultModelLoader(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        # For source side, allow optional download.
        self._default_loader.download_model(model_config)

    def _get_extra(self) -> dict[str, Any]:
        extra = self.load_config.model_loader_extra_config or {}
        if not isinstance(extra, dict):
            raise ValueError("model_loader_extra_config must be a dict")
        return extra

    def _register_to_coordinator(
        self,
        *,
        role: str,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        my_id: str,
        npu_id: int,
        params: list[dict[str, Any]],
    ) -> dict:
        extra = self._get_extra()
        coord = extra.get("coordinator_url")
        if not coord:
            raise ValueError("model_loader_extra_config.coordinator_url is required")
        node_ip = _resolve_node_ip(extra)
        rank_info = _get_rank_info()
        payload = {
            "role": role,
            "model_key": _build_model_key(vllm_config, model_config),
            "node_ip": node_ip,
            "my_id": my_id,
            "npu_id": npu_id,
            "rank_info": rank_info,
            "params": params,
        }
        return _http_post_json(f"{coord}/v1/registry/register", payload)

    def _poll_tasks(
        self,
        *,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        my_id: str,
    ) -> list[dict[str, Any]]:
        extra = self._get_extra()
        coord = extra.get("coordinator_url")
        payload = {
            "role": "source",
            "model_key": _build_model_key(vllm_config, model_config),
            "my_id": my_id,
            "rank_info": _get_rank_info(),
        }
        resp = _http_post_json(f"{coord}/v1/registry/poll", payload)
        return resp.get("tasks", [])

    def _wait_done(
        self,
        *,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        my_id: str,
        timeout_s: int,
    ) -> None:
        extra = self._get_extra()
        coord = extra.get("coordinator_url")
        start = time.time()
        while True:
            payload = {
                "role": "receiver",
                "model_key": _build_model_key(vllm_config, model_config),
                "my_id": my_id,
                "rank_info": _get_rank_info(),
            }
            resp = _http_post_json(f"{coord}/v1/registry/wait", payload, timeout_s=10)
            if resp.get("status") == "done":
                return
            if time.time() - start > timeout_s:
                raise TimeoutError("wait for transfer done timed out")
            time.sleep(float(extra.get("poll_interval_s", 2)))

    def _transfer_tasks(
        self,
        engine: Any,
        tasks: list[dict[str, Any]],
        local_params: dict[str, dict[str, Any]],
    ) -> None:
        for task in tasks:
            peer_id = task.get("peer_id")
            if not peer_id:
                logger.warning("task missing peer_id, skip")
                continue
            dst_params = task.get("dst_params", {})
            transfer_id = task.get("transfer_id")
            for name, meta in local_params.items():
                if name not in dst_params:
                    continue
                dst = dst_params[name]
                src_addr = int(meta["addr"])
                dst_addr = int(dst["addr"])
                size = int(meta["bytes"])
                ret = engine.transfer_sync_write(peer_id, src_addr, dst_addr, size)
                if ret != 0:
                    logger.error("transfer failed ret=%s name=%s", ret, name)
                    raise RuntimeError(f"transfer failed ret={ret} name={name}")
            if transfer_id:
                extra = self._get_extra()
                coord = extra.get("coordinator_url")
                payload = {"transfer_id": transfer_id}
                _http_post_json(f"{coord}/v1/registry/complete", payload)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        extra = self._get_extra()
        role = str(extra.get("role", "source")).lower()
        if role not in ("source", "receiver"):
            raise ValueError("model_loader_extra_config.role must be source or receiver")

        vllm_config = getattr(self, "_vllm_config", None)
        if vllm_config is None:
            raise RuntimeError("vllm_config is not set")

        rank_info = _get_rank_info()
        node_ip = _resolve_node_ip(extra)
        my_id = _build_my_id(extra, node_ip, rank_info["rank"])
        npu_id = _get_npu_id(extra)
        params = _params_metadata(model)
        local_params = {p["name"]: p for p in params}

        memfabric_role = extra.get("memfabric_role")
        if memfabric_role is None:
            memfabric_role = "Prefill" if role == "source" else "Decode"
        engine = _initialize_engine(extra, my_id, npu_id, memfabric_role)
        _register_memory(engine, params)
        try:
            torch.npu.synchronize()
        except Exception:
            pass

        self._register_to_coordinator(
            role=role,
            vllm_config=vllm_config,
            model_config=model_config,
            my_id=my_id,
            npu_id=npu_id,
            params=params,
        )

        if role == "receiver":
            timeout_s = int(extra.get("poll_timeout_s", 1800))
            self._wait_done(
                vllm_config=vllm_config,
                model_config=model_config,
                my_id=my_id,
                timeout_s=timeout_s,
            )
            return

        # role == source
        poll_interval = float(extra.get("poll_interval_s", 2))
        timeout_s = int(extra.get("poll_timeout_s", 1800))

        def _run():
            start = time.time()
            while True:
                tasks = self._poll_tasks(
                    vllm_config=vllm_config, model_config=model_config, my_id=my_id
                )
                if tasks:
                    self._transfer_tasks(engine, tasks, local_params)
                if time.time() - start > timeout_s:
                    return
                time.sleep(poll_interval)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        self._vllm_config = vllm_config
        role = str(self._get_extra().get("role", "source")).lower()
        if role == "source":
            # source uses default loader to read weights
            model = self._default_loader.load_model(vllm_config, model_config)
            # after weights loaded, do memfabric register/transfer
            self.load_weights(model, model_config)
            return model
        # receiver: initialize model and wait for transfer
        return super().load_model(vllm_config, model_config)
