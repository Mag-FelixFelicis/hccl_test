# Copyright (c) 2026
# Minimal HTTP control plane for memfabric weight transfer

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STATE: dict[str, Any] = {
    "models": {},
    "next_transfer_id": 1,
}
LOCK = threading.Lock()


def _model_key_str(model_key: dict) -> str:
    return json.dumps(model_key, sort_keys=True, separators=(",", ":"))


def _rank_key(rank_info: dict) -> str:
    tp = int(rank_info.get("tp_rank", 0))
    pp = int(rank_info.get("pp_rank", 0))
    dp = int(rank_info.get("dp_rank", 0))
    return f"tp:{tp}|pp:{pp}|dp:{dp}"


def _get_model_state(key: str) -> dict[str, Any]:
    models = STATE["models"]
        if key not in models:
            models[key] = {
                "sources": {},
                "receivers": {},
                "pending": {},
                "assignments": {},
                "source_assignments": {},
                "transfer_status": {},
                "receiver_transfers": {},
                "ready_sources": set(),
                "ready_receivers": set(),
            }
    return models[key]


def _new_transfer_id() -> str:
    tid = STATE["next_transfer_id"]
    STATE["next_transfer_id"] += 1
    return f"t{tid}"


def _params_to_map(params: list[dict]) -> dict[str, dict[str, Any]]:
    out = {}
    for p in params:
        name = p.get("name")
        if not name:
            continue
        out[name] = {
            "addr": int(p.get("addr", 0)),
            "bytes": int(p.get("bytes", 0)),
        }
    return out


def _maybe_create_tasks(model_state: dict, rank_key: str):
    # only dispatch when both sides are ready
    source_ready = any(
        key.startswith(f"{rank_key}|") for key in model_state.get("ready_sources", set())
    )
    receiver_ready = any(
        key.startswith(f"{rank_key}|")
        for key in model_state.get("ready_receivers", set())
    )
    if not (source_ready and receiver_ready):
        return
    source = model_state["sources"].get(rank_key)
    if not source:
        return
    receivers = model_state["receivers"].get(rank_key, {})
    for rid, recv in receivers.items():
        if recv.get("transfer_id"):
            continue
        transfer_id = _new_transfer_id()
        task = {
            "transfer_id": transfer_id,
            "peer_id": recv["my_id"],
            "dst_params": recv["params_map"],
        }
        model_state.setdefault("pending", {}).setdefault(source["my_id"], []).append(
            task
        )
        model_state["transfer_status"][transfer_id] = "pending"
        model_state.setdefault("receiver_transfers", {}).setdefault(recv["my_id"], []).append(
            transfer_id
        )
        recv["transfer_id"] = transfer_id


class Handler(BaseHTTPRequestHandler):
    server_version = "memfabric-coord/0.1"

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/registry/assign":
            self._handle_assign()
            return
        if self.path == "/v1/registry/register":
            self._handle_register()
            return
        if self.path == "/v1/registry/ready":
            self._handle_ready()
            return
        if self.path == "/v1/registry/poll":
            self._handle_poll()
            return
        if self.path == "/v1/registry/complete":
            self._handle_complete()
            return
        if self.path == "/v1/registry/wait":
            self._handle_wait()
            return
        self._send_json(404, {"error": "not found"})

    def _handle_assign(self):
        req = self._read_json()
        model_key = req.get("model_key", {})
        my_id = req.get("my_id")
        rank_info = req.get("rank_info", {})
        if not my_id:
            self._send_json(400, {"error": "missing my_id"})
            return

        key = _model_key_str(model_key)
        rank_key = _rank_key(rank_info)
        with LOCK:
            state = _get_model_state(key)
            role = state["assignments"].get(my_id)
            if role is None:
                assigned = state.get("source_assignments", {}).get(rank_key)
                if assigned and assigned != my_id:
                    role = "receiver"
                else:
                    role = "source"
                    state.setdefault("source_assignments", {})[rank_key] = my_id
                state["assignments"][my_id] = role
        self._send_json(200, {"role": role})

    def _handle_register(self):
        req = self._read_json()
        model_key = req.get("model_key", {})
        my_id = req.get("my_id")
        role = req.get("role")
        rank_info = req.get("rank_info", {})
        params = req.get("params", [])
        if not my_id:
            self._send_json(400, {"error": "missing my_id"})
            return

        key = _model_key_str(model_key)
        rank_key = _rank_key(rank_info)
        params_map = _params_to_map(params)
        metrics = req.get("metrics", {})
        with LOCK:
            state = _get_model_state(key)
            if role is None:
                role = state["assignments"].get(my_id, "source")
            if role == "source":
                state.setdefault("source_assignments", {})[rank_key] = my_id
                state["sources"][rank_key] = {
                    "my_id": my_id,
                    "rank_info": rank_info,
                    "params_map": params_map,
                    "metrics": metrics,
                    "ts": time.time(),
                }
                _maybe_create_tasks(state, rank_key)
            else:
                state.setdefault("receivers", {}).setdefault(rank_key, {})[my_id] = {
                    "my_id": my_id,
                    "rank_info": rank_info,
                    "params_map": params_map,
                    "metrics": metrics,
                    "ts": time.time(),
                }
                _maybe_create_tasks(state, rank_key)

        self._send_json(200, {"status": "ok", "role": role})

    def _handle_ready(self):
        req = self._read_json()
        model_key = req.get("model_key", {})
        my_id = req.get("my_id")
        role = req.get("role")
        rank_info = req.get("rank_info", {})
        if not my_id:
            self._send_json(400, {"error": "missing my_id"})
            return
        key = _model_key_str(model_key)
        rank_key = _rank_key(rank_info)
        with LOCK:
            state = _get_model_state(key)
            if role is None:
                role = state["assignments"].get(my_id)
            if role == "source":
                state["ready_sources"].add(f"{rank_key}|{my_id}")
            else:
                state["ready_receivers"].add(f"{rank_key}|{my_id}")
            # only create tasks if both sides are ready
            if role in ("source", "receiver"):
                source_ready = any(
                    key.startswith(f"{rank_key}|") for key in state["ready_sources"]
                )
                receiver_ready = any(
                    key.startswith(f"{rank_key}|")
                    for key in state["ready_receivers"]
                )
                if source_ready and receiver_ready:
                    _maybe_create_tasks(state, rank_key)
        self._send_json(200, {"status": "ok"})

    def _handle_poll(self):
        req = self._read_json()
        model_key = req.get("model_key", {})
        my_id = req.get("my_id")
        if not my_id:
            self._send_json(400, {"error": "missing my_id"})
            return

        key = _model_key_str(model_key)
        with LOCK:
            state = _get_model_state(key)
            tasks = state.get("pending", {}).get(my_id, [])
            state.get("pending", {})[my_id] = []
        self._send_json(200, {"tasks": tasks})

    def _handle_complete(self):
        req = self._read_json()
        transfer_id = req.get("transfer_id")
        if not transfer_id:
            self._send_json(400, {"error": "missing transfer_id"})
            return
        with LOCK:
            for _, state in STATE["models"].items():
                if transfer_id in state.get("transfer_status", {}):
                    state["transfer_status"][transfer_id] = "done"
                    break
        self._send_json(200, {"status": "ok"})

    def _handle_wait(self):
        req = self._read_json()
        model_key = req.get("model_key", {})
        my_id = req.get("my_id")
        if not my_id:
            self._send_json(400, {"error": "missing my_id"})
            return
        key = _model_key_str(model_key)
        with LOCK:
            state = _get_model_state(key)
            transfers = state.get("receiver_transfers", {}).get(my_id, [])
            if not transfers:
                self._send_json(200, {"status": "wait"})
                return
            done = True
            for tid in transfers:
                if state.get("transfer_status", {}).get(tid) != "done":
                    done = False
                    break
        self._send_json(200, {"status": "done" if done else "wait"})


def main():
    parser = argparse.ArgumentParser(description="MemFabric HTTP Coordinator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"coordinator listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
