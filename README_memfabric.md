# MemFabric 跨节点显存传输测速（1GiB）

本示例使用 **MemFabric TRANS + SHM 控制面** 实现两节点 D2D 传输测速，并在接收端读取显存数据校验。

- 2 个节点、每节点 1 张 NPU
- 1GiB Tensor，形状为 `[4096, 65536]`（float32）
- rank0 发送、rank1 接收并校验显存数据

## 前置条件

两台节点都已安装并 `source`：

```bash
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

确认 MemFabric 安装路径：

```bash
echo $MEMFABRIC_HYBRID_HOME_PATH
ls $MEMFABRIC_HYBRID_HOME_PATH
```

安装目录里通常有 `${arch}-${os}` 子目录，例如 `aarch64-linux` 或 `x86_64-linux`。

## 编译

在两台节点都编译一次：

```bash
ARCH_DIR=<你的arch-os目录, 例如 aarch64-linux>

g++ -std=c++14 -O2 memfabric_trans_bench.cpp \
  -I$MEMFABRIC_HYBRID_HOME_PATH/$ARCH_DIR/include/smem/host \
  -I$ASCEND_HOME_PATH/include \
  -L$MEMFABRIC_HYBRID_HOME_PATH/$ARCH_DIR/lib64 \
  -L$ASCEND_HOME_PATH/lib64 \
  -lmf_smem -lascendcl \
  -o memfabric_trans_bench
```

## 运行（两节点）

准备：选择一个 **rank0 节点IP和端口** 作为 config store，例如 `192.168.201.14:8570`。

### 节点A（rank0）

```bash
./memfabric_trans_bench \
  --rank 0 --world 2 --device 0 \
  --store-url tcp://192.168.201.14:8570 \
  --my-id 192.168.201.14:10001 \
  --peer-id 192.168.201.15:10001 \
  --bytes 1073741824 --warmup 1 --iters 5
```

### 节点B（rank1）

```bash
./memfabric_trans_bench \
  --rank 1 --world 2 --device 0 \
  --store-url tcp://192.168.201.14:8570 \
  --my-id 192.168.201.15:10001 \
  --peer-id 192.168.201.14:10001 \
  --bytes 1073741824 --warmup 1 --iters 5
```

说明：
- `--store-url` 必须指向 rank0 节点的 IP:PORT
- `--my-id` / `--peer-id` 用于 TRANS 的唯一标识（推荐用 `ip:port`）
- `--bytes` 默认是 1GiB，可调整

## 结果与校验

rank0 输出平均耗时与带宽：
```
avg_ms=... throughput=... GiB/s (... GB/s)
```

rank1 输出显存数据校验结果（头/尾 8 个元素）：
```
verify_head=OK verify_tail=OK
```

## 机制说明（为什么“另一个进程能读到显存数据”）

本示例中，rank1 进程是独立进程，它在本进程内分配并注册了 **自己的显存地址**，  
rank0 通过 TRANS 写入 rank1 的显存，因此 **rank1 进程可以直接读取并校验**。

这不是“跨进程直接读取同一指针”，而是 **跨节点传输到对端显存**，对端进程读取本地显存。
