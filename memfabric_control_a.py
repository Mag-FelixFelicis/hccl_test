#!/usr/bin/env python3
# coding=utf-8

import argparse
import socket
import time

import torch
import torch_npu  # noqa: F401
from memfabric_hybrid import TransferEngine, create_config_store, set_log_level, set_conf_store_tls


def parse_args():
    p = argparse.ArgumentParser(description="Process A: create tensor and send on command")
    p.add_argument("--store-url", required=True, help="tcp://ip:port for config store")
    p.add_argument("--my-id", required=True, help="unique id for this node, e.g. ip:port")
    p.add_argument("--peer-id", required=True, help="unique id for peer, e.g. ip:port")
    p.add_argument("--npu-id", type=int, default=0)
    p.add_argument("--listen-ip", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=9000)
    p.add_argument("--bytes", type=int, default=1 << 30)
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
    create_config_store(args.store_url)
    time.sleep(2)

    engine = TransferEngine()
    # TransferEngine expects role "Prefill" or "Decode"
    ret = engine.initialize(
        args.store_url,
        args.my_id,
        "Prefill",
        args.npu_id,
        TransferEngine.TransDataOpType.DEVICE_RDMA,
    )
    if ret != 0:
        raise RuntimeError("TransferEngine initialize failed")

    tensor = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32, device="npu").reshape(shape)
    total_bytes = tensor.element_size() * tensor.numel()
    engine.register_memory(tensor.data_ptr(), total_bytes)
    torch.npu.synchronize()

    print(f"[A] tensor shape={shape} bytes={total_bytes}")
    print(f"[A] dev_addr={hex(tensor.data_ptr())}")
    print(f"[A] first_row_head={tensor[0, :8].cpu().tolist()}")

    b_addr = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.listen_ip, args.listen_port))
        s.listen(5)
        print(f"[A] control server listening {args.listen_ip}:{args.listen_port}")
        while True:
            conn, _ = s.accept()
            with conn:
                data = conn.recv(4096).decode("utf-8").strip()
                if data.startswith("REG "):
                    b_addr = int(data.split()[1], 16)
                    print(f"[A] received B dev_addr={hex(b_addr)}")
                    conn.sendall(b"OK\n")
                elif data.startswith("SEND"):
                    if b_addr is None:
                        conn.sendall(b"ERR no B addr\n")
                        continue
                    print("[A] send command received, start D2D transfer")
                    engine.transfer_sync_write(args.peer_id, tensor.data_ptr(), b_addr, total_bytes)
                    torch.npu.synchronize()
                    print("[A] transfer done")
                    conn.sendall(b"OK\n")
                else:
                    conn.sendall(b"ERR unknown\n")


if __name__ == "__main__":
    main()
