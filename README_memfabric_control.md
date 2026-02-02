# MemFabric 三进程穿刺示例（A/B/C）

目标：
- 进程 A（节点A）创建 1GiB tensor 到 NPU 显存，打印地址和第一行
- 进程 B（节点B）接收 A 的 D2D 传输，打印显存地址和第一行
- 进程 C 向 A 发指令触发传输

文件：
- `memfabric_control_a.py`：A 进程
- `memfabric_control_b.py`：B 进程
- `memfabric_control_c.py`：C 进程

## 前置条件

两台节点都安装并 source：

```bash
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

确保 Python 可用：

```bash
python3 -c "import memfabric_hybrid, torch, torch_npu; print('ok')"
```

## 运行步骤

假设：
- 节点A IP：`192.168.201.14`
- 节点B IP：`192.168.201.15`
- config store 使用 `tcp://192.168.201.14:8570`
- A 控制端口 `9000`
- 唯一ID使用 `ip:port`

### 1) 节点A启动进程A

```bash
python3 memfabric_control_a.py \
  --store-url tcp://192.168.201.14:8570 \
  --my-id 192.168.201.14:10001 \
  --peer-id 192.168.201.15:10001 \
  --npu-id 0 \
  --listen-ip 0.0.0.0 --listen-port 9000 \
  --b-notify-ip 192.168.201.15 --b-notify-port 9001 \
  --bytes 1073741824
```

### 2) 节点B启动进程B

```bash
python3 memfabric_control_b.py \
  --store-url tcp://192.168.201.14:8570 \
  --my-id 192.168.201.15:10001 \
  --npu-id 0 \
  --notify-ip 192.168.201.14 --notify-port 9000 \
  --listen-ip 0.0.0.0 --listen-port 9001 \
  --bytes 1073741824
```

### 3) 任意机器启动进程C发送指令

```bash
python3 memfabric_control_c.py --ip 192.168.201.14 --port 9000 --cmd SEND
```

## 预期输出

进程A：
- 以 Prefill 角色初始化 TransferEngine
- 打印显存地址与第一行
- 收到 C 指令后执行 D2D 传输

进程B：
- 以 Decode 角色初始化 TransferEngine
- 打印显存地址
- 传输完成后打印第一行（应为 1..）

## 说明

- 进程B将自己的显存地址通过 TCP 发给进程A（A 作为控制面）
- 进程C只是触发 A 进行传输
- 这是“穿刺原型”，用于验证传输与指令流程

- NOTE: this sample uses DEVICE_RDMA; ensure A2 device RDMA is enabled and device network is reachable.
