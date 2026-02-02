# MemFabric 三进程穿刺示例（A/B/C�?
目标�?- 进程 A（节点A）创�?1GiB int32 tensor �?NPU 显存，打印地址和第一�?- 进程 B（节点B）接�?A �?D2D 传输，打印显存地址和第一�?- 进程 C �?A 发指令触发传�?
文件�?- `memfabric_control_a.py`：A 进程
- `memfabric_control_b.py`：B 进程
- `memfabric_control_c.py`：C 进程

## 前置条件

两台节点都安装并 source�?
```bash
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

确保 Python 可用�?
```bash
python3 -c "import memfabric_hybrid, torch, torch_npu; print('ok')"
```

## 运行步骤

假设�?- 节点A IP：`192.168.201.14`
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

### 3) 任意机器启动进程C发送指�?
```bash
python3 memfabric_control_c.py --ip 192.168.201.14 --port 9000 --cmd SEND
```

## 预期输出

进程A�?- �?Prefill 角色初始�?TransferEngine
- 打印显存地址与第一�?- 收到 C 指令后执�?D2D 传输
- 传输前发�?START，传输后发�?DONE <ms> �?B

进程B�?- �?Decode 角色初始�?TransferEngine
- 打印显存地址
- 收到 START 后开始计时，收到 DONE 后读取并打印

## 说明

- 进程B将自己的显存地址通过 TCP 发给进程A（A 作为控制面）
- 进程C只是触发 A 进行传输
- 这是“穿刺原型”，用于验证传输与指令流�?
- NOTE: this sample uses DEVICE_RDMA; ensure A2 device RDMA is enabled and device network is reachable.
