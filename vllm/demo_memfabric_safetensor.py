#!/usr/bin/env python3
# Minimal memfabric demo: disk safetensor -> NPU -> D2D to another NPU

import argparse
import json
import os
import socket
import time
import urllib.request
from typing import Any

import torch
import torch_npu  # noqa: F401


def http_post_json(url: str, payload: dict[str, Any], timeout_s: int = 5) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    if not body:
        return {}
    return json.loads(body)


def resolve_node_ip(explicit: str | None) -> str:
    if explicit:
        return explicit
    for key in ("POD_IP", "HOST_IP", "VLLM_NODE_IP"):
        if key in os.environ:
            return os.environ[key]
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def get_rank() -> int:
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def init_engine(store_url: str, my_id: str, role: str, npu_id: int, log_level: int):
    from memfabric_hybrid import (
        TransferEngine,
        create_config_store,
        set_conf_store_tls,
        set_log_level,
    )

    set_log_level(log_level)
    set_conf_store_tls(False, "")
    create_config_store(store_url)
    time.sleep(1)
    engine = TransferEngine()
    ret = engine.initialize(
        store_url, my_id, role, npu_id, TransferEngine.TransDataOpType.DEVICE_RDMA
    )
    if ret != 0:
        raise RuntimeError(f"TransferEngine initialize failed ret={ret}")
    return engine


def expected_last_row_tail(rows: int, cols: int) -> list[int]:
    start = (rows - 1) * cols + (cols - 8) + 1
    return [int(start + i) for i in range(8)]


def main():
    p = argparse.ArgumentParser(description="MemFabric minimal safetensor demo")
    p.add_argument("--coordinator-url", required=True)
    p.add_argument("--store-url", required=True)
    p.add_argument("--node-ip", default=None)
    p.add_argument("--base-port", type=int, default=10000)
    p.add_argument("--npu-id", type=int, default=0)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=65536)
    p.add_argument("--dtype", default="int32")
    p.add_argument("--safetensor-path", default="/tmp/demo_tensor.safetensors")
    p.add_argument("--log-level", type=int, default=1)
    p.add_argument("--poll-interval-s", type=float, default=2)
    p.add_argument("--poll-timeout-s", type=int, default=1800)
    args = p.parse_args()

    node_ip = resolve_node_ip(args.node_ip)
    rank = get_rank()
    my_id = f"{node_ip}:{args.base_port + rank}"
    shape = (args.rows, args.cols)
    dtype = getattr(torch, args.dtype)

    model_key = {
        "name": "demo_safetensor_4096x65536",
        "shape": list(shape),
        "dtype": args.dtype,
    }
    assign = http_post_json(
        f"{args.coordinator_url}/v1/registry/assign",
        {"model_key": model_key, "my_id": my_id, "node_ip": node_ip, "rank_info": {"rank": rank}},
    )
    role = str(assign.get("role", "source")).lower()
    if role not in ("source", "receiver"):
        role = "source"

    engine_role = "Prefill" if role == "source" else "Decode"
    engine = init_engine(args.store_url, my_id, engine_role, args.npu_id, args.log_level)

    if role == "source":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("safetensors is required for this demo") from e

        if not os.path.exists(args.safetensor_path):
            raise FileNotFoundError(
                f"{args.safetensor_path} not found. Run create_safetensor.py first."
            )

        t0 = time.perf_counter()
        loaded = load_file(args.safetensor_path, device="cpu")
        cpu_tensor = loaded["demo"]
        npu_tensor = cpu_tensor.to("npu")
        torch.npu.synchronize()
        t1 = time.perf_counter()
        load_ms = (t1 - t0) * 1000.0
        bytes_size = npu_tensor.numel() * npu_tensor.element_size()
        gib = bytes_size / (1024.0 * 1024.0 * 1024.0)
        gibps = gib / ((t1 - t0) if (t1 - t0) > 0 else 1e-6)
        print(f"[source] disk->NPU ms={load_ms:.3f} throughput={gibps:.2f} GiB/s")

        tail = npu_tensor[-1, -8:].cpu().tolist()
        expect = expected_last_row_tail(args.rows, args.cols)
        print(f"[source] last_row_tail={tail} expected={expect} ok={tail == expect}")

        engine.register_memory(npu_tensor.data_ptr(), bytes_size)

        payload = {
            "role": "source",
            "model_key": model_key,
            "my_id": my_id,
            "node_ip": node_ip,
            "rank_info": {"rank": rank},
            "params": [
                {
                    "name": "demo",
                    "dtype": args.dtype,
                    "shape": list(shape),
                    "numel": int(npu_tensor.numel()),
                    "bytes": int(bytes_size),
                    "addr": int(npu_tensor.data_ptr()),
                }
            ],
            "metrics": {"disk_to_npu_ms": load_ms, "disk_to_npu_gibps": gibps},
        }
        http_post_json(f"{args.coordinator_url}/v1/registry/register", payload)

        print("[source] waiting for receiver tasks...")
        start = time.time()
        while True:
            tasks = http_post_json(
                f"{args.coordinator_url}/v1/registry/poll",
                {"model_key": model_key, "my_id": my_id, "rank_info": {"rank": rank}},
            ).get("tasks", [])
            for task in tasks:
                peer_id = task.get("peer_id")
                dst_params = task.get("dst_params", {})
                if "demo" not in dst_params:
                    continue
                dst_addr = int(dst_params["demo"]["addr"])
                t2 = time.perf_counter()
                ret = engine.transfer_sync_write(peer_id, npu_tensor.data_ptr(), dst_addr, bytes_size)
                torch.npu.synchronize()
                t3 = time.perf_counter()
                if ret != 0:
                    raise RuntimeError(f"transfer failed ret={ret}")
                ms = (t3 - t2) * 1000.0
                gibps = gib / ((t3 - t2) if (t3 - t2) > 0 else 1e-6)
                print(f"[source] transfer ms={ms:.3f} throughput={gibps:.2f} GiB/s")
                if task.get("transfer_id"):
                    http_post_json(
                        f"{args.coordinator_url}/v1/registry/complete",
                        {"transfer_id": task["transfer_id"]},
                    )
            if time.time() - start > args.poll_timeout_s:
                break
            time.sleep(args.poll_interval_s)

    else:
        npu_tensor = torch.empty(shape, dtype=dtype, device="npu")
        bytes_size = npu_tensor.numel() * npu_tensor.element_size()
        engine.register_memory(npu_tensor.data_ptr(), bytes_size)

        payload = {
            "role": "receiver",
            "model_key": model_key,
            "my_id": my_id,
            "node_ip": node_ip,
            "rank_info": {"rank": rank},
            "params": [
                {
                    "name": "demo",
                    "dtype": args.dtype,
                    "shape": list(shape),
                    "numel": int(npu_tensor.numel()),
                    "bytes": int(bytes_size),
                    "addr": int(npu_tensor.data_ptr()),
                }
            ],
        }
        http_post_json(f"{args.coordinator_url}/v1/registry/register", payload)

        print("[receiver] waiting for transfer...")
        start = time.perf_counter()
        while True:
            resp = http_post_json(
                f"{args.coordinator_url}/v1/registry/wait",
                {"model_key": model_key, "my_id": my_id, "rank_info": {"rank": rank}},
            )
            if resp.get("status") == "done":
                break
            if (time.perf_counter() - start) > args.poll_timeout_s:
                raise TimeoutError("wait timeout")
            time.sleep(args.poll_interval_s)

        torch.npu.synchronize()
        end = time.perf_counter()
        wait_ms = (end - start) * 1000.0
        gib = bytes_size / (1024.0 * 1024.0 * 1024.0)
        gibps = gib / ((end - start) if (end - start) > 0 else 1e-6)
        print(f"[receiver] transfer ms={wait_ms:.3f} throughput={gibps:.2f} GiB/s")
        tail = npu_tensor[-1, -8:].cpu().tolist()
        expect = expected_last_row_tail(args.rows, args.cols)
        print(f"[receiver] last_row_tail={tail} expected={expect} ok={tail == expect}")


if __name__ == "__main__":
    main()
