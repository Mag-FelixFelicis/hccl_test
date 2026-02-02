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
    p.add_argument("--listen-ip", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=9001)
    p.add_argument("--log-level", type=int, default=1, choices=[0, 1, 2, 3])
    return p.parse_args()


def main():
    args = parse_args()
    if args.bytes % (4096 * 4) != 0:
        raise ValueError("bytes must be divisible by 4096 * sizeof(int32)")
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

    tensor = torch.zeros(shape, dtype=torch.int32, device="npu")
    total_bytes = tensor.element_size() * tensor.numel()
    engine.register_memory(tensor.data_ptr(), total_bytes)
    torch.npu.synchronize()

    print(f"[B] tensor shape={shape} bytes={total_bytes}")
    print(f"[B] dev_addr={hex(tensor.data_ptr())}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.notify_ip, args.notify_port))
        s.sendall(f"REG {hex(tensor.data_ptr())}\n".encode("utf-8"))
        _ = s.recv(128)

    print("[B] registered addr sent to A, waiting for START/DONE notify...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.listen_ip, args.listen_port))
        s.listen(5)
        idx = 0
        start_ts = None
        while True:
            conn, _ = s.accept()
            with conn:
                msg = conn.recv(64).decode("utf-8").strip()
            if msg.startswith("START"):
                start_ts = time.time()
                print(f"[B] recv#{idx} start")
                continue
            if msg.startswith("DONE"):
                done_ms = None
                parts = msg.split()
                if len(parts) == 2:
                    try:
                        done_ms = float(parts[1])
                    except ValueError:
                        done_ms = None
                if start_ts is None:
                    start_ts = time.time()
                torch.npu.synchronize()
                head = tensor[0, :8].cpu()
                tail = tensor.view(-1)[-8:].cpu()
                ok_tail = (tail[0].item() == (tensor.numel() - 7))
                elapsed_ms = (time.time() - start_ts) * 1000.0
                print(f"[B] recv#{idx} first_row_head={head.tolist()}")
                print(f"[B] recv#{idx} last_row_tail={tail.tolist()} tail_ok={ok_tail}")
                print(f"[B] recv#{idx} bytes={total_bytes} recv_time_ms={elapsed_ms:.3f}")
                if done_ms is not None:
                    print(f"[B] recv#{idx} sender_ms={done_ms:.3f}")
                idx += 1
                start_ts = None


if __name__ == "__main__":
    main()
