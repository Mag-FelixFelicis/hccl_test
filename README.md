# HCCL 2-Node 1-Card P2P Send/Recv Benchmark

This repo provides a simple HCCL point-to-point test:

- 2 nodes
- 1 NPU per node
- 1 GiB payload (default)
- Rank0 sends, Rank1 receives
- Reports average time and throughput

The program uses `HcclSend`/`HcclRecv` and a shared `rootInfo` file for
cross-node initialization.

## Prerequisites

- Ascend CANN installed on both nodes
- HCCL/ACL runtime available
- Two nodes can reach each other (for HCCL)

Load CANN environment (example default path):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## Build

Run on both nodes:

```bash
g++ -std=c++14 -O2 test.cpp \
  -I$ASCEND_HOME_PATH/include \
  -L$ASCEND_HOME_PATH/lib64 \
  -lhccl -lascendcl \
  -o hccl_p2p
```

## Run (2 nodes)

### Step 1: Start Rank0 (sender) on Node A

```bash
./hccl_p2p --rank 0 --world 2 --device 0 \
  --root-info /tmp/rootinfo.bin \
  --iters 20 --warmup 3 --bytes 1073741824
```

This writes `/tmp/rootinfo.bin`. Copy it to Node B, or use a shared path.

### Step 2: Start Rank1 (receiver) on Node B

```bash
./hccl_p2p --rank 1 --world 2 --device 0 \
  --root-info /tmp/rootinfo.bin \
  --iters 20 --warmup 3 --bytes 1073741824
```

## Output

Each rank prints:

```
rank=<id> bytes=<n> avg_ms=<ms> throughput=<GiB/s> (<GB/s>)
```

## Parameters

- `--rank`     : 0 or 1
- `--world`    : total ranks (fixed to 2)
- `--device`   : local NPU id (usually 0)
- `--root-info`: path to shared rootInfo file
- `--bytes`    : payload size in bytes (default 1 GiB)
- `--warmup`   : warmup iterations
- `--iters`    : timed iterations

## Notes

- If you prefer decimal 1GB, use `--bytes 1000000000`.
- Rank1 waits up to 300s for the rootInfo file.
- Make sure both nodes run with the same `--bytes` and `--iters`.
