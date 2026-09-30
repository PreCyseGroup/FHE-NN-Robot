"""Inference server for Kanayama, plaintext NN, and CKKS NN control."""

import json
import math
import os
import socket
import struct
import time
from pathlib import Path

import numpy as np

try:
    import tenseal as ts
except ImportError:
    ts = None

HOST = os.environ.get("AMR_HOST", "0.0.0.0")
PORT = int(os.environ.get("AMR_PORT", "50007"))
MODE = os.environ.get("AMR_MODE", "auto").lower()
MODEL_PATH = Path(os.environ.get("AMR_MODEL", "model_export.json"))
OUTPUT_DIR = Path(os.environ.get("AMR_SAVE", "."))

N_INPUTS = 5
KX, KY, KTH = 2.0, 9.0, 6.0
WHEEL_RADIUS = 0.04445
WHEELBASE = 0.393
WHEEL_SPEED_LIMIT = 10.0
VALID_MODES = ("kanayama", "plaintext", "ckks")


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        data += chunk
    return data


def recv_frame(sock):
    (length,) = struct.unpack("!I", recv_exact(sock, 4))
    return recv_exact(sock, length)


def send_frame(sock, payload):
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def load_model(path):
    with path.open("r", encoding="utf-8") as f:
        export = json.load(f)

    fc1_w = np.asarray(export["fc1_weight"], dtype=np.float64)
    if fc1_w.ndim != 2 or fc1_w.shape[0] != N_INPUTS:
        raise RuntimeError(
            f"Expected a {N_INPUTS}-input model, but fc1_weight has shape "
            f"{fc1_w.shape}. Retrain with the new train.py."
        )

    return {
        "fc1_weight": export["fc1_weight"],
        "fc1_bias": export["fc1_bias"],
        "fc3_weight": export["fc3_weight"],
        "fc3_bias": export["fc3_bias"],
        "fc1_w": fc1_w,
        "fc1_b": np.asarray(export["fc1_bias"], dtype=np.float64),
        "fc3_w": np.asarray(export["fc3_weight"], dtype=np.float64),
        "fc3_b": np.asarray(export["fc3_bias"], dtype=np.float64),
    }


def clip_wheel(value):
    return float(np.clip(value, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT))


def kanayama_from_features(features):
    if len(features) != N_INPUTS:
        raise ValueError(
            f"Kanayama mode expects {N_INPUTS} features, got {len(features)}"
        )

    ex, ey, eth, v_ref, omega_ref = map(float, features)
    v_cmd = v_ref * math.cos(eth) + KX * ex
    omega_cmd = omega_ref + v_ref * (KY * ey + KTH * math.sin(eth))
    wr = (v_cmd + 0.5 * WHEELBASE * omega_cmd) / WHEEL_RADIUS
    wl = (v_cmd - 0.5 * WHEELBASE * omega_cmd) / WHEEL_RADIUS
    return clip_wheel(wr), clip_wheel(wl)


def plaintext_forward(x, model):
    z1 = x @ model["fc1_w"] + model["fc1_b"]
    a1 = z1 + 0.125 * z1 * z1
    return a1 @ model["fc3_w"] + model["fc3_b"]


def compute_response(mode, payload, model, context):
    start = time.perf_counter()

    if mode == "ckks":
        enc_x = ts.ckks_vector_from(context, payload)
        enc_z1 = enc_x.matmul(model["fc1_weight"]) + model["fc1_bias"]
        enc_a1 = enc_z1 + enc_z1.square() * 0.125
        enc_y = enc_a1.matmul(model["fc3_weight"]) + model["fc3_bias"]
        response = enc_y.serialize()

    else:
        features = json.loads(payload)["features"]
        if mode == "kanayama":
            wr, wl = kanayama_from_features(features)
            response = json.dumps({"wr": wr, "wl": wl}).encode()
        else:
            if len(features) != N_INPUTS:
                raise ValueError(
                    f"Plaintext NN expects {N_INPUTS} features, got {len(features)}"
                )
            y = plaintext_forward(np.asarray(features, dtype=np.float64), model)
            response = json.dumps({"y": [float(y[0]), float(y[1])]}).encode()

    return response, (time.perf_counter() - start) * 1000.0


def save_processing_times(mode, times_ms):
    if not times_ms:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"server_processing_times_{mode}.npz"
    np.savez(path, processing_time_ms=np.asarray(times_ms), mode=mode)
    print(f"[Server] Saved timing data: {path}")


def handle_client(conn, model):
    hello = json.loads(recv_frame(conn))
    requested_mode = str(hello.get("mode", "ckks")).lower()
    mode = requested_mode if MODE == "auto" else MODE

    if mode not in VALID_MODES:
        send_frame(conn, json.dumps({"ok": False, "error": "invalid mode"}).encode())
        return

    if mode in ("plaintext", "ckks") and model is None:
        raise RuntimeError(f"Model file is required for {mode} mode: {MODEL_PATH}")

    send_frame(conn, json.dumps({"ok": True, "mode": mode}).encode())
    print(f"[Server] Client session started in {mode} mode.")

    context = None
    if mode == "ckks":
        if ts is None:
            raise RuntimeError("TenSEAL is required for CKKS mode")
        print("[Server] Receiving CKKS public context...")
        context = ts.context_from(recv_frame(conn))
        send_frame(conn, json.dumps({"ok": True}).encode())
        print("[Server] CKKS context ready. Waiting for inference requests...")

    times_ms = []
    request_count = 0
    try:
        while True:
            try:
                payload = recv_frame(conn)
            except ConnectionError:
                break

            response, elapsed_ms = compute_response(mode, payload, model, context)
            times_ms.append(elapsed_ms)
            send_frame(conn, response)
            request_count += 1

            if request_count % 100 == 0:
                print(f"[Server] Processed {request_count} requests ({mode}).")
    finally:
        print(f"[Server] Session finished after {request_count} requests.")
        save_processing_times(mode, times_ms)


def main():
    if MODE not in VALID_MODES + ("auto",):
        raise SystemExit("AMR_MODE must be auto, kanayama, plaintext, or ckks")

    model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
    if model is not None:
        print(f"[Server] Model loaded from {MODEL_PATH}")
    else:
        print(
            f"[Server] Model file not found at {MODEL_PATH}; "
            "Kanayama mode is still available."
        )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT} (mode={MODE})")

        while True:
            conn, address = server_sock.accept()
            print(f"[Server] Connection accepted from {address[0]}:{address[1]}")
            with conn:
                try:
                    handle_client(conn, model)
                except Exception as exc:
                    print(f"[Server] Session error: {exc}")


if __name__ == "__main__":
    main()
