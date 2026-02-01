# MemFabric Python 跨节点显存传输测速（1GiB）

本示例使用 **MemFabric TransferEngine(Python)** 在两节点间做 D2D 传输测速，并在接收端读取显存数据校验。

- 2 个节点、每节点 1 张 NPU
- 1GiB Tensor，形状 `[4096, 65536]`（float32）
- Sender 写入 Receiver 的显存地址

## 前置条件

两台节点都已安装并 `source`：

```bash
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

确保 Python 能 import：

```bash
python3 -c "import memfabric_hybrid, torch, torch_npu; print('ok')"
```

## 运行（两节点）

选择一个 rank0 节点 IP 和端口作为 config store，例如 `192.168.201.14:8570`。

### 节点A（Sender）

```bash
python3 memfabric_trans_bench.py \
  --role Sender \
  --store-url tcp://192.168.201.14:8570 \
  --my-id 192.168.201.14:10001 \
  --peer-id 192.168.201.15:10001 \
  --npu-id 0 \
  --bytes 1073741824 --warmup 1 --iters 5
```

### 节点B（Receiver）

```bash
python3 memfabric_trans_bench.py \
  --role Receiver \
  --store-url tcp://192.168.201.14:8570 \
  --my-id 192.168.201.15:10001 \
  --peer-id 192.168.201.14:10001 \
  --npu-id 0 \
  --bytes 1073741824
```

## 输出

Sender 会输出平均耗时与带宽：
```
avg_ms=... throughput=... GiB/s (... GB/s)
```

Receiver 会输出显存数据校验结果：
```
verify_head=OK verify_tail=OK
```

## 说明

- Python 版本基于 MemFabric 的 `TransferEngine`，与 C++ 示例逻辑一致。
- Receiver 进程读取的是 **本进程显存**（由 Sender 传输写入），而不是跨进程直接访问同一指针。
