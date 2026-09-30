"""Webots client for Kanayama, plaintext NN, and CKKS NN control.

The client runs a 50 s Lissajous tracking experiment with a 100 ms control
period. Remote inference is performed in a background thread and the most
recent wheel command is held until the next response arrives.

The controller input is:
    [e_x, e_y, e_theta, v_ref, omega_ref]
"""

import json
import os
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
from controller import Supervisor

try:
    import tenseal as ts
except ImportError:
    ts = None


N_INPUTS = 5
WHEEL_RADIUS = 0.04445
WHEELBASE = 0.393
WHEEL_SPEED_LIMIT = 10.0
CONTROL_DT = 0.1
SIM_TIMESTEP_MS = 10
CONTROL_PERIOD_MS = 100
MAX_SIM_TIME = 50.0
N_WARMUP_REQUESTS = 2

MODE = os.environ.get("AMR_MODE", "ckks").lower()  # kanayama | plaintext | ckks
SERVER_HOST = os.environ.get("AMR_SERVER", "127.0.0.1")
SERVER_PORT = int(os.environ.get("AMR_PORT", "50007"))

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("AMR_MODEL", PROJECT_DIR / "model_export.json"))
SAVE_DIR = Path(os.environ.get("AMR_SAVE", PROJECT_DIR))

DRAW_REFERENCE_PATH = True
REFERENCE_PATH_Z = 0.006
REFERENCE_PATH_WIDTH = 0.030
REFERENCE_PATH_STRIDE = 2


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def clip_wheels(wr, wl):
    return (
        float(np.clip(wr, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)),
        float(np.clip(wl, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)),
    )


def make_lissajous_reference():
    t = np.arange(0.0, MAX_SIM_TIME + CONTROL_DT, CONTROL_DT)
    tau = 2.0 * np.pi * t / MAX_SIM_TIME
    x = 0.8 * np.sin(3.0 * tau + np.pi / 2.0)
    y = 0.8 * np.sin(2.0 * tau)

    xd = np.gradient(x, t, edge_order=2)
    yd = np.gradient(y, t, edge_order=2)
    xdd = np.gradient(xd, t, edge_order=2)
    ydd = np.gradient(yd, t, edge_order=2)
    speed = np.sqrt(xd**2 + yd**2)
    theta = np.unwrap(np.arctan2(yd, xd))
    omega = (xd * ydd - yd * xdd) / (speed**2 + 1e-6)
    return t, x, y, theta, speed, omega


class TrajectoryReference:
    def __init__(self):
        self.t, self.x, self.y, self.theta, self.speed, self.omega = (
            make_lissajous_reference()
        )

    def query(self, t):
        t = min(float(t), float(self.t[-1]))
        theta = np.arctan2(
            np.interp(t, self.t, np.sin(self.theta)),
            np.interp(t, self.t, np.cos(self.theta)),
        )
        return (
            np.interp(t, self.t, self.x),
            np.interp(t, self.t, self.y),
            theta,
            np.interp(t, self.t, self.speed),
            np.interp(t, self.t, self.omega),
        )


def tracking_features(x, y, theta, reference, t):
    x_ref, y_ref, theta_ref, v_ref, omega_ref = reference.query(t)
    dx = x_ref - x
    dy = y_ref - y
    return np.array(
        [
            np.cos(theta) * dx + np.sin(theta) * dy,
            -np.sin(theta) * dx + np.cos(theta) * dy,
            wrap_to_pi(theta_ref - theta),
            v_ref,
            omega_ref,
        ],
        dtype=np.float64,
    )


def add_reference_path(robot, reference):
    """Draw the Lissajous reference trajectory on the Webots floor."""
    if not DRAW_REFERENCE_PATH:
        return

    old_path = robot.getFromDef("REFERENCE_TRAJECTORY")
    if old_path is not None:
        old_path.remove()

    center = [
        [float(reference.x[i]), float(reference.y[i]), REFERENCE_PATH_Z]
        for i in range(0, len(reference.t), REFERENCE_PATH_STRIDE)
    ]
    last_point = [
        float(reference.x[-1]),
        float(reference.y[-1]),
        REFERENCE_PATH_Z,
    ]
    if center[-1] != last_point:
        center.append(last_point)
    center = np.asarray(center, dtype=np.float64)

    tangents = np.zeros((len(center), 2), dtype=np.float64)
    tangents[0] = center[1, :2] - center[0, :2]
    tangents[-1] = center[-1, :2] - center[-2, :2]
    if len(center) > 2:
        tangents[1:-1] = center[2:, :2] - center[:-2, :2]

    lengths = np.linalg.norm(tangents, axis=1)
    lengths[lengths < 1e-12] = 1.0
    normals = np.column_stack(
        (-(tangents / lengths[:, None])[:, 1], (tangents / lengths[:, None])[:, 0])
    )

    half_width = REFERENCE_PATH_WIDTH / 2.0
    vertices = []
    for point, normal in zip(center, normals):
        left = point.copy()
        right = point.copy()
        left[:2] += half_width * normal
        right[:2] -= half_width * normal
        vertices.extend((left, right))

    faces = [
        (2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2)
        for i in range(len(center) - 1)
    ]

    point_text = ",\n".join(
        f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices
    )
    face_text = ",\n".join(
        f"{a}, {b}, {c}, {d}, -1" for a, b, c, d in faces
    )

    node = f"""
    DEF REFERENCE_TRAJECTORY Shape {{
      appearance PBRAppearance {{
        baseColor 1 0 0
        roughness 0.7
        metalness 0
      }}
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [
            {point_text}
          ]
        }}
        coordIndex [
          {face_text}
        ]
        solid FALSE
      }}
      castShadows FALSE
      isPickable FALSE
    }}
    """
    robot.getRoot().getField("children").importMFNodeFromString(-1, node)


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


def make_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 30, 30, 30, 30, 40],
    )
    context.global_scale = 2**30
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


class AsyncRemoteController:
    def __init__(
        self,
        sock,
        mode,
        reference,
        context=None,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
    ):
        self.sock = sock
        self.mode = mode
        self.reference = reference
        self.context = context
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self.wheel_speeds = (0.0, 0.0)
        self.failure = None

        self.control_update_t = []
        self.wheel_speed_log = []
        self.round_trip_times_ms = []
        self.request_preparation_times_ms = []
        self.network_server_times_ms = []
        self.response_processing_times_ms = []

    def start(self):
        self._thread.start()

    def stop(self):
        self._queue.join()
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5.0)

    def get_wheel_speeds(self):
        with self._lock:
            return self.wheel_speeds

    def schedule(self, x, y, theta, control_t):
        self._queue.put((x, y, theta, control_t))

    def _prepare_features(self, x, y, theta, t):
        raw = tracking_features(x, y, theta, self.reference, t)
        if self.mode == "kanayama":
            return raw, None
        return raw, (raw - self.x_mean) / self.x_std

    def _round_trip(self, raw, normalized):
        t0 = time.perf_counter()

        if self.mode == "ckks":
            payload = ts.ckks_vector(self.context, normalized.tolist()).serialize()
        elif self.mode == "plaintext":
            payload = json.dumps({"features": normalized.tolist()}).encode()
        else:
            payload = json.dumps({"features": raw.tolist()}).encode()

        t1 = time.perf_counter()
        send_frame(self.sock, payload)
        response = recv_frame(self.sock)
        t2 = time.perf_counter()

        if self.mode == "ckks":
            out = np.asarray(
                ts.ckks_vector_from(self.context, response).decrypt()[:2],
                dtype=np.float64,
            )
            wr, wl = out * self.y_std + self.y_mean
        elif self.mode == "plaintext":
            out = np.asarray(json.loads(response)["y"], dtype=np.float64)
            wr, wl = out * self.y_std + self.y_mean
        else:
            out = json.loads(response)
            wr, wl = float(out["wr"]), float(out["wl"])

        t3 = time.perf_counter()
        return (
            wr,
            wl,
            (t3 - t0) * 1000.0,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
        )

    def _record(self, t, wr, wl, rtt, prep, network_server, response):
        self.control_update_t.append(float(t))
        self.wheel_speed_log.append((wr, wl))
        self.round_trip_times_ms.append(rtt)
        self.request_preparation_times_ms.append(prep)
        self.network_server_times_ms.append(network_server)
        self.response_processing_times_ms.append(response)

    def warmup(self, x, y, theta, t):
        raw, normalized = self._prepare_features(x, y, theta, t)
        for _ in range(N_WARMUP_REQUESTS):
            self._round_trip(raw, normalized)

    def prime(self, x, y, theta, t):
        raw, normalized = self._prepare_features(x, y, theta, t)
        wr, wl, rtt, prep, net, response = self._round_trip(raw, normalized)
        wr, wl = clip_wheels(wr, wl)
        with self._lock:
            self.wheel_speeds = (wr, wl)
        self._record(t, wr, wl, rtt, prep, net, response)
        return wr, wl

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            x, y, theta, t = item
            try:
                raw, normalized = self._prepare_features(x, y, theta, t)
                wr, wl, rtt, prep, net, response = self._round_trip(raw, normalized)
                wr, wl = clip_wheels(wr, wl)

                with self._lock:
                    self.wheel_speeds = (wr, wl)

                self._record(t, wr, wl, rtt, prep, net, response)
            except Exception as exc:
                self.failure = exc
                with self._lock:
                    self.wheel_speeds = (0.0, 0.0)
            finally:
                self._queue.task_done()

            if self.failure is not None:
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self._queue.task_done()
                return


def load_normalization():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run train.py first.")

    with MODEL_PATH.open("r", encoding="utf-8") as f:
        export = json.load(f)

    x_mean = np.asarray(export["x_mean"], dtype=np.float64)
    x_std = np.asarray(export["x_std"], dtype=np.float64)
    y_mean = np.asarray(export["y_mean"], dtype=np.float64)
    y_std = np.asarray(export["y_std"], dtype=np.float64)

    if len(x_mean) != N_INPUTS or len(x_std) != N_INPUTS:
        raise SystemExit(
            f"Expected normalization for {N_INPUTS} inputs, but model contains "
            f"{len(x_mean)}. Retrain with the new 5-input train.py."
        )

    return x_mean, x_std, y_mean, y_std


def connect_to_server(requested_mode):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_HOST, SERVER_PORT))
    except OSError as exc:
        sock.close()
        raise SystemExit(f"Cannot connect to inference server: {exc}") from exc

    send_frame(sock, json.dumps({"mode": requested_mode}).encode())
    reply = json.loads(recv_frame(sock))
    if not reply.get("ok"):
        sock.close()
        raise SystemExit(f"Server rejected the session: {reply}")
    return sock, str(reply["mode"]).lower()


def main():
    if MODE not in ("kanayama", "plaintext", "ckks"):
        raise SystemExit("AMR_MODE must be kanayama, plaintext, or ckks")

    reference = TrajectoryReference()
    print(
        f"[Client] Connecting to {SERVER_HOST}:{SERVER_PORT} "
        f"(requested mode: {MODE})..."
    )
    sock, mode = connect_to_server(MODE)
    print(f"[Client] Connected. Active mode: {mode}")

    context = None
    if mode == "ckks":
        if ts is None:
            raise SystemExit("TenSEAL is required for CKKS mode")
        print("[Client] Creating CKKS context...")
        context = make_ckks_context()
        public_context = context.serialize(save_secret_key=False)
        send_frame(sock, public_context)
        ack = json.loads(recv_frame(sock))
        if not ack.get("ok"):
            raise SystemExit("Server rejected the CKKS context")
        print("[Client] CKKS context sent to server.")

    x_mean = x_std = y_mean = y_std = None
    if mode in ("plaintext", "ckks"):
        x_mean, x_std, y_mean, y_std = load_normalization()
        print(f"[Client] Loaded normalization values from {MODEL_PATH}.")

    robot = Supervisor()
    add_reference_path(robot, reference)
    print("[Client] Reference trajectory added to the Webots world.")

    node = robot.getSelf()
    translation = node.getField("translation")
    rotation = node.getField("rotation")

    x0, y0, theta0, _, _ = reference.query(0.0)
    z0 = translation.getSFVec3f()[2]
    translation.setSFVec3f([x0, y0, z0])
    rotation.setSFRotation([0.0, 0.0, 1.0, theta0 % (2.0 * np.pi)])
    node.resetPhysics()

    right_motor = robot.getDevice("motor_r")
    left_motor = robot.getDevice("motor_l")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    gps = robot.getDevice("gps")
    compass = robot.getDevice("compass")
    gps.enable(SIM_TIMESTEP_MS)
    compass.enable(SIM_TIMESTEP_MS)

    controller = AsyncRemoteController(
        sock,
        mode,
        reference,
        context,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    print(f"[Client] Running {N_WARMUP_REQUESTS} warm-up requests...")
    controller.warmup(x0, y0, theta0, 0.0)
    print("[Client] Warm-up complete. Starting 50 s evaluation...")

    wr0, wl0 = controller.prime(x0, y0, theta0, 0.0)
    right_motor.setVelocity(wr0)
    left_motor.setVelocity(wl0)
    controller.start()

    next_control_index = 1
    actual_positions = []
    reference_positions = []
    next_progress_time = 10.0

    while robot.step(SIM_TIMESTEP_MS) != -1:
        if controller.failure is not None:
            raise RuntimeError(f"Inference request failed: {controller.failure}")

        pos = gps.getValues()
        x, y = float(pos[0]), float(pos[1])
        north = compass.getValues()
        theta = float(np.arctan2(north[0], north[1]))

        sim_time = float(robot.getTime())
        while (
            next_control_index < len(reference.t)
            and sim_time + 1e-9 >= reference.t[next_control_index]
        ):
            controller.schedule(
                x,
                y,
                theta,
                float(reference.t[next_control_index]),
            )
            next_control_index += 1

        wr, wl = controller.get_wheel_speeds()
        right_motor.setVelocity(wr)
        left_motor.setVelocity(wl)

        actual_positions.append((x, y))
        x_ref, y_ref, _, _, _ = reference.query(sim_time)
        reference_positions.append((x_ref, y_ref))

        if sim_time + 1e-9 >= next_progress_time:
            print(
                f"[Client] Simulation progress: "
                f"{min(next_progress_time, MAX_SIM_TIME):.0f}/{MAX_SIM_TIME:.0f} s; "
                f"completed control requests: "
                f"{len(controller.round_trip_times_ms)}/{len(reference.t)}"
            )
            next_progress_time += 10.0

        if sim_time >= MAX_SIM_TIME:
            break

    right_motor.setVelocity(0.0)
    left_motor.setVelocity(0.0)
    print("[Client] Simulation finished. Waiting for pending inference requests...")
    controller.stop()
    sock.close()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SAVE_DIR / f"run_data_webots_{mode}.npz"
    np.savez(
        output_path,
        traj_name="lissajous",
        mode=mode,
        control_period_ms=CONTROL_PERIOD_MS,
        sim_timestep_ms=SIM_TIMESTEP_MS,
        actual_positions=np.asarray(actual_positions),
        reference_positions=np.asarray(reference_positions),
        control_update_t=np.asarray(controller.control_update_t),
        wheel_speeds=np.asarray(controller.wheel_speed_log),
        round_trip_times_ms=np.asarray(controller.round_trip_times_ms),
        request_preparation_times_ms=np.asarray(
            controller.request_preparation_times_ms
        ),
        network_server_times_ms=np.asarray(controller.network_server_times_ms),
        response_processing_times_ms=np.asarray(
            controller.response_processing_times_ms
        ),
    )
    print(f"Saved run data: {output_path}")


if __name__ == "__main__":
    main()
