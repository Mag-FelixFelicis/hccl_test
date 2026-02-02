#!/usr/bin/env python3
# coding=utf-8

import argparse
import socket
import time

import torch
import torch_npu  # noqa: F401
from memfabric_hybrid import TransferEngine, set_log_level, set_conf_store_tls


def parse_args():
    p = argparse.ArgumentParser(description="Process B: receive tensor and print first row")
    p.add_argument("--store-url", required=True, help="tcp://ip:port for config store")
    p.add_argument("--my-id", required=True, help="unique id for this node, e.g. ip:port")
    p.add_argument("--npu-id", type=int, default=0)
    p.add_argument("--bytes", type=int, default=1 << 30)
    p.add_argument("--notify-ip", required=True, help="A control server ip")
    p.add_argument("--notify-port", type=int, default=9000)
    p.add_argument("--log-level", type=int, default=1, choices=[0, 1, 2, 3])
    return p.parse_args()


def main():
    args = parse_args()
    if args.bytes % (4096 * 4) != 0:
        raise ValueError("bytes must be divisible by 4096 * sizeof(float32)")
    cols = args.bytes // (4096 * 4)
    shape = (4096, cols)

    set_log_level(args.log_level)
    set_conf_store_tls(False, "")

    engine = TransferEngine()
    # TransferEngine expects role "Prefill" or "Decode"
    ret = engine.initialize(
        args.store_url,
        args.my_id,
        "Decode",
        args.npu_id,
        TransferEngine.TransDataOpType.DEVICE_RDMA,
    )
    if ret != 0:
        raise RuntimeError("TransferEngine initialize failed")

    tensor = torch.zeros(shape, dtype=torch.float32, device="npu")
    total_bytes = tensor.element_size() * tensor.numel()
    engine.register_memory(tensor.data_ptr(), total_bytes)
    torch.npu.synchronize()

    print(f"[B] tensor shape={shape} bytes={total_bytes}")
    print(f"[B] dev_addr={hex(tensor.data_ptr())}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.notify_ip, args.notify_port))
        s.sendall(f"REG {hex(tensor.data_ptr())}\n".encode("utf-8"))
        _ = s.recv(128)

    print("[B] registered addr sent to A, waiting for transfer...")
    # Poll until the first element becomes 1.0 or timeout
    timeout_s = 120
    start = time.time()
    while True:
        torch.npu.synchronize()
        head = tensor[0, :8].cpu()
        if head[0].item() == 1.0:
            elapsed_ms = (time.time() - start) * 1000.0
            # verify tail to confirm full size
            tail = tensor.view(-1)[-8:].cpu()
            ok_tail = (tail[0].item() == (tensor.numel() - 7))
            print(f"[B] first_row_head={head.tolist()}")
            print(f"[B] last_row_tail={tail.tolist()} tail_ok={ok_tail}")
            print(f"[B] recv_bytes={total_bytes} recv_time_ms={elapsed_ms:.3f}")
            break
        if time.time() - start > timeout_s:
            print("[B] timeout waiting for data, first_row_head=", head.tolist())
            break
        time.sleep(1)

    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
