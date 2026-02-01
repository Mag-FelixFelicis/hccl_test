#!/usr/bin/env python3
# coding=utf-8

import argparse
import time

import torch
import torch_npu  # noqa: F401
from memfabric_hybrid import TransferEngine, create_config_store, set_log_level, set_conf_store_tls


def parse_args():
    parser = argparse.ArgumentParser(description="MemFabric TRANS 1GiB D2D benchmark")
    parser.add_argument("--role", required=True, choices=["Sender", "Receiver"])
    parser.add_argument("--store-url", required=True, help="tcp://ip:port for config store (rank0)")
    parser.add_argument("--my-id", required=True, help="unique id for this node, e.g. ip:port")
    parser.add_argument("--peer-id", required=True, help="unique id for peer, e.g. ip:port")
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument("--bytes", type=int, default=1 << 30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--log-level", type=int, default=1, choices=[0, 1, 2, 3])
    return parser.parse_args()


def verify_head_tail(tensor, k=8):
    head = tensor.view(-1)[:k].cpu()
    tail = tensor.view(-1)[-k:].cpu()
    head_expected = torch.arange(1, k + 1, dtype=torch.float32)
    tail_expected = torch.arange(tensor.numel() - k + 1, tensor.numel() + 1, dtype=torch.float32)
    ok1 = torch.allclose(head, head_expected)
    ok2 = torch.allclose(tail, tail_expected)
    print(f"verify_head={'OK' if ok1 else 'FAIL'} verify_tail={'OK' if ok2 else 'FAIL'}")


def main():
    args = parse_args()
    set_log_level(args.log_level)
    set_conf_store_tls(False, "")

    if args.bytes % (4096 * 4) != 0:
        raise ValueError("bytes must be divisible by 4096 * sizeof(float32)")
    cols = args.bytes // (4096 * 4)
    shape = (4096, cols)
    print(f"shape={shape} bytes={args.bytes}")

    engine = TransferEngine()

    if args.role == "Sender":
        create_config_store(args.store_url)
        time.sleep(2)

    ret = engine.initialize(args.store_url, args.my_id, args.role, args.npu_id)
    if ret != 0:
        raise RuntimeError("TransferEngine initialize failed")

    if args.role == "Receiver":
        tensor = torch.zeros(shape, dtype=torch.float32, device="npu")
        total_bytes = tensor.element_size() * tensor.numel()
        engine.register_memory(tensor.data_ptr(), total_bytes)
        print(f"receiver registered addr={hex(tensor.data_ptr())} bytes={total_bytes}")

        # wait for sender to finish transfer
        time.sleep(5)
        torch.npu.synchronize()
        verify_head_tail(tensor)
        while True:
            time.sleep(10)

    else:
        tensor = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32, device="npu").reshape(shape)
        total_bytes = tensor.element_size() * tensor.numel()
        engine.register_memory(tensor.data_ptr(), total_bytes)
        print(f"sender registered addr={hex(tensor.data_ptr())} bytes={total_bytes}")
        time.sleep(5)

        for _ in range(args.warmup):
            engine.transfer_sync_write(args.peer_id, tensor.data_ptr(), tensor.data_ptr(), total_bytes)
        torch.npu.synchronize()

        total_ms = 0.0
        for _ in range(args.iters):
            t0 = time.time()
            engine.transfer_sync_write(args.peer_id, tensor.data_ptr(), tensor.data_ptr(), total_bytes)
            torch.npu.synchronize()
            t1 = time.time()
            total_ms += (t1 - t0) * 1000.0

        avg_ms = total_ms / args.iters
        gib = total_bytes / (1024.0 * 1024.0 * 1024.0)
        gb = total_bytes / 1e9
        gibps = gib / (avg_ms / 1000.0)
        gbps = gb / (avg_ms / 1000.0)
        print(f"avg_ms={avg_ms:.3f} throughput={gibps:.2f} GiB/s ({gbps:.2f} GB/s)")


if __name__ == "__main__":
    main()
