#!/usr/bin/env python3
# Create a safetensors file with incremental values.

import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser(description="Create safetensor with incremental values")
    p.add_argument("--path", default="/tmp/demo_tensor.safetensors")
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=65536)
    p.add_argument("--dtype", default="int32")
    args = p.parse_args()

    try:
        from safetensors.torch import save_file
    except Exception as e:
        raise RuntimeError("safetensors is required to run this script") from e

    dtype = getattr(torch, args.dtype)
    shape = (args.rows, args.cols)
    numel = shape[0] * shape[1]

    tensor = torch.arange(1, numel + 1, dtype=dtype).reshape(shape)
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    save_file({"demo": tensor}, args.path)
    size_bytes = os.path.getsize(args.path)

    tail = tensor[-1, -8:].tolist()
    print(f"saved: {args.path}")
    print(f"shape={shape} dtype={args.dtype} bytes={size_bytes}")
    print(f"last_row_tail={tail}")


if __name__ == "__main__":
    main()
