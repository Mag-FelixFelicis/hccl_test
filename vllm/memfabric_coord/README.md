MemFabric HTTP Coordinator
=========================

This folder contains a minimal HTTP control plane for memfabric D2D weight
transfer and example Kubernetes manifests + startup scripts.

Key idea
--------
All vLLM pods use the same load format:

  --load-format memfabric_http

The loader calls the coordinator to get a role (source or receiver). The first
pod for a given (model, tp, pp, dtype, quant, revision) becomes a source. Later
pods become receivers and wait for D2D transfers.

Coordinator API
---------------
POST /v1/registry/assign
POST /v1/registry/register
POST /v1/registry/poll
POST /v1/registry/complete
POST /v1/registry/wait

See coordinator.py for exact request/response shapes.

Run locally
-----------
python3 coordinator.py --host 0.0.0.0 --port 8080

Demo (minimal, no vLLM)
-----------------------
Scripts:
- ../create_safetensor.py
- ../demo_memfabric_safetensor.py

Step 1: create safetensor on disk (run once, source node)
python3 ../create_safetensor.py \
  --path /tmp/demo_tensor.safetensors \
  --rows 4096 --cols 65536 --dtype int32

Source node:
python3 ../demo_memfabric_safetensor.py \
  --coordinator-url http://127.0.0.1:8080 \
  --store-url tcp://127.0.0.1:8570 \
  --node-ip 192.168.201.14 \
  --safetensor-path /tmp/demo_tensor.safetensors

Receiver node:
python3 ../demo_memfabric_safetensor.py \
  --coordinator-url http://127.0.0.1:8080 \
  --store-url tcp://127.0.0.1:8570 \
  --node-ip 192.168.201.15 \
  --safetensor-path /tmp/demo_tensor.safetensors

Note:
- The demo script will start the config store ONLY on the source role.
- The receiver will not start a local store server and will only connect to the
  existing store on the source node.

vLLM startup (all pods use the same args)
----------------------------------------
Example extra config JSON:

{
  "coordinator_url": "http://coordinator:8080",
  "store_url": "tcp://coordinator:8570",
  "node_ip": "192.168.201.14",
  "base_port": 10000,
  "poll_interval_s": 2,
  "poll_timeout_s": 1800,
  "log_level": 1
}

Notes
-----
1) node_ip can be set via env var: POD_IP, HOST_IP, or VLLM_NODE_IP.
2) store_url is the memfabric config store address. In practice, choose a
   stable Service IP/port and make sure the store is reachable by all pods.
