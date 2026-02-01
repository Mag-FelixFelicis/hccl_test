#!/usr/bin/env python3
# coding=utf-8

import argparse
import socket


def parse_args():
    p = argparse.ArgumentParser(description="Process C: send control command to A")
    p.add_argument("--ip", required=True, help="A control server ip")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--cmd", choices=["SEND"], default="SEND")
    return p.parse_args()


def main():
    args = parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.ip, args.port))
        s.sendall(f"{args.cmd}\n".encode("utf-8"))
        resp = s.recv(128).decode("utf-8").strip()
        print(resp)


if __name__ == "__main__":
    main()
