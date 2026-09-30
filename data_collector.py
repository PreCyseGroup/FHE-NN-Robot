"""Collect Webots imitation-learning data from the Kanayama controller.

The saved dataset contains:
    features = [e_x, e_y, e_theta, v_ref, omega_ref]
    targets  = [omega_r, omega_l]

Data collection is designed for reproducibility and broad state-space coverage:
- trajectory generation uses its own RNG;
- rollout start phases are evenly distributed, not randomly selected;
- initial tracking-error perturbations come from a seeded Sobol sequence;
- each rollout is short and focused on recovery behavior.

The controller runs every 100 ms while Webots advances with a 10 ms step.
"""

import os
from pathlib import Path

import numpy as np
from controller import Supervisor
from scipy.interpolate import CubicSpline
from scipy.stats import qmc


SEED = 42
TRAJECTORY_SEED = SEED
PERTURBATION_SEED = SEED + 1

WHEEL_RADIUS = 0.04445
WHEELBASE = 0.393
WHEEL_SPEED_LIMIT = 10.0
CONTROL_DT = 0.1
SIM_TIMESTEP_MS = 10
STEPS_PER_CONTROL = int(round(CONTROL_DT / (SIM_TIMESTEP_MS / 1000.0)))
KX, KY, KTH = 2.0, 9.0, 6.0

TRAJECTORY_NAMES = [
    "lemniscate",
    "circle",
    "rounded_square",
    "spiral",
    "rose",
    "clover",
    "hypotrochoid",
    "random_spline",
]

# Many short recovery rollouts are more useful than a few long rollouts.
N_START_POINTS = 16
N_PERTURBATIONS = 32  # Keep this a power of two for Sobol random_base2().
ROLLOUT_STEPS = 40    # 4.0 s at CONTROL_DT = 0.1 s.

# Initial tracking-error ranges.
EX_PERTURB = 0.30
EY_PERTURB = 0.30
HEADING_PERTURB = 0.70

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = Path(os.environ.get("AMR_DATASET", PROJECT_DIR / "webots_dataset.npz"))

TRAJECTORY_DURATIONS = {
    "lemniscate": None,
    "circle": 42.0,
    "rounded_square": 48.0,
    "spiral": 55.0,
    "rose": 90.0,
    "clover": 48.0,
    "hypotrochoid": 75.0,
    "random_spline": 60.0,
}


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def clip_wheels(wr, wl):
    return (
        float(np.clip(wr, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)),
        float(np.clip(wl, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)),
    )


def unicycle_to_wheel(v, omega):
    wr = (v + 0.5 * WHEELBASE * omega) / WHEEL_RADIUS
    wl = (v - 0.5 * WHEELBASE * omega) / WHEEL_RADIUS
    return wr, wl


def make_trajectory(name, rng):
    duration = TRAJECTORY_DURATIONS[name]

    if name == "lemniscate":
        alpha, eta = 5.0, 0.8
        t = np.arange(0.0, 4.0 * np.pi * alpha + CONTROL_DT, CONTROL_DT)
        x = eta * np.sin(t / alpha)
        y = eta * np.sin(t / (2.0 * alpha))

    elif name == "circle":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        tau = 2.0 * np.pi * t / duration
        x, y = 0.8 * np.cos(tau), 0.8 * np.sin(tau)

    elif name == "rounded_square":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        exponent = 3.2
        tau_dense = np.linspace(0.0, 2.0 * np.pi, 200_000)
        x_dense = (
            0.85
            * np.sign(np.cos(tau_dense))
            * np.abs(np.cos(tau_dense)) ** (2.0 / exponent)
        )
        y_dense = (
            0.85
            * np.sign(np.sin(tau_dense))
            * np.abs(np.sin(tau_dense)) ** (2.0 / exponent)
        )
        ds = np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        tau = np.interp((t / duration) * s[-1], s, tau_dense)
        x = (
            0.85
            * np.sign(np.cos(tau))
            * np.abs(np.cos(tau)) ** (2.0 / exponent)
        )
        y = (
            0.85
            * np.sign(np.sin(tau))
            * np.abs(np.sin(tau)) ** (2.0 / exponent)
        )

    elif name == "spiral":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        tau = np.linspace(0.0, 6.0 * np.pi, len(t))
        radius = np.linspace(0.1, 0.8, len(t))
        x, y = radius * np.cos(tau), radius * np.sin(tau)

    elif name == "rose":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        tau = 2.0 * np.pi * t / duration
        radius = 0.8 * np.cos(5.0 * tau)
        x, y = radius * np.cos(tau), radius * np.sin(tau)

    elif name == "clover":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        tau = 2.0 * np.pi * t / duration
        radius = 0.6 + 0.2 * np.cos(4.0 * tau)
        x, y = radius * np.cos(tau), radius * np.sin(tau)

    elif name == "hypotrochoid":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        tau = np.linspace(0.0, 8.0 * np.pi, len(t))
        R, r, d, scale = 5.0, 3.0, 4.0, 0.12
        x = scale * (
            (R - r) * np.cos(tau) + d * np.cos((R - r) / r * tau)
        )
        y = scale * (
            (R - r) * np.sin(tau) - d * np.sin((R - r) / r * tau)
        )

    elif name == "random_spline":
        t = np.arange(0.0, duration + CONTROL_DT, CONTROL_DT)
        radius = rng.uniform(0.3, 0.8, 10)
        angle = np.sort(rng.uniform(0.0, 2.0 * np.pi, 10))
        x_wp = np.append(radius * np.cos(angle), radius[0] * np.cos(angle[0]))
        y_wp = np.append(radius * np.sin(angle), radius[0] * np.sin(angle[0]))
        s_wp = np.linspace(0.0, 1.0, len(x_wp))
        s = np.linspace(0.0, 1.0, len(t))
        x = CubicSpline(s_wp, x_wp, bc_type="periodic")(s)
        y = CubicSpline(s_wp, y_wp, bc_type="periodic")(s)

    else:
        raise ValueError(f"Unknown trajectory: {name}")

    xd = np.gradient(x, t, edge_order=2)
    yd = np.gradient(y, t, edge_order=2)
    xdd = np.gradient(xd, t, edge_order=2)
    ydd = np.gradient(yd, t, edge_order=2)
    speed = np.sqrt(xd**2 + yd**2)
    theta = np.unwrap(np.arctan2(yd, xd))
    omega = (xd * ydd - yd * xdd) / (speed**2 + 1e-6)

    return {
        "t": t,
        "x": x,
        "y": y,
        "theta": theta,
        "speed": speed,
        "omega": omega,
    }


def make_perturbations():
    """Create deterministic, well-spread initial tracking-error samples."""
    if N_PERTURBATIONS <= 0 or (N_PERTURBATIONS & (N_PERTURBATIONS - 1)) != 0:
        raise ValueError("N_PERTURBATIONS must be a positive power of two")

    m = int(np.log2(N_PERTURBATIONS))
    sampler = qmc.Sobol(d=3, scramble=True, seed=PERTURBATION_SEED)
    unit = sampler.random_base2(m=m)
    perturbations = qmc.scale(
        unit,
        [-EX_PERTURB, -EY_PERTURB, -HEADING_PERTURB],
        [EX_PERTURB, EY_PERTURB, HEADING_PERTURB],
    )

    # Guarantee that the nominal state is represented exactly.
    perturbations[0] = np.array([0.0, 0.0, 0.0])
    return perturbations


def pose_from_tracking_error(trajectory, k, ex, ey, eth):
    """Construct a robot pose that has the requested Kanayama tracking errors."""
    theta_ref = float(trajectory["theta"][k])
    theta = theta_ref - float(eth)

    c = np.cos(theta)
    s = np.sin(theta)

    # [ex, ey]^T = R(-theta) * ([x_ref, y_ref] - [x, y])
    dx = c * ex - s * ey
    dy = s * ex + c * ey

    x = float(trajectory["x"][k] - dx)
    y = float(trajectory["y"][k] - dy)
    return x, y, theta


def tracking_features(x, y, theta, trajectory, k):
    dx = trajectory["x"][k] - x
    dy = trajectory["y"][k] - y
    return np.array(
        [
            np.cos(theta) * dx + np.sin(theta) * dy,
            -np.sin(theta) * dx + np.cos(theta) * dy,
            wrap_to_pi(trajectory["theta"][k] - theta),
            trajectory["speed"][k],
            trajectory["omega"][k],
        ],
        dtype=np.float32,
    )


def kanayama_controller(x, y, theta, trajectory, k):
    dx = trajectory["x"][k] - x
    dy = trajectory["y"][k] - y
    ex = np.cos(theta) * dx + np.sin(theta) * dy
    ey = -np.sin(theta) * dx + np.cos(theta) * dy
    eth = wrap_to_pi(trajectory["theta"][k] - theta)
    v_ref = trajectory["speed"][k]
    omega_ref = trajectory["omega"][k]

    v_cmd = v_ref * np.cos(eth) + KX * ex
    omega_cmd = omega_ref + v_ref * (KY * ey + KTH * np.sin(eth))
    return clip_wheels(*unicycle_to_wheel(v_cmd, omega_cmd))


def read_pose(gps, compass):
    pos = gps.getValues()
    north = compass.getValues()
    return pos[0], pos[1], float(np.arctan2(north[0], north[1]))


def step_control_period(robot):
    for _ in range(STEPS_PER_CONTROL):
        if robot.step(SIM_TIMESTEP_MS) == -1:
            return False
    return True


def main():
    trajectory_rng = np.random.default_rng(TRAJECTORY_SEED)
    perturbations = make_perturbations()

    robot = Supervisor()
    robot_node = robot.getSelf()
    translation = robot_node.getField("translation")
    rotation = robot_node.getField("rotation")
    z0 = translation.getSFVec3f()[2]

    right_motor = robot.getDevice("motor_r")
    left_motor = robot.getDevice("motor_l")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    gps = robot.getDevice("gps")
    compass = robot.getDevice("compass")
    gps.enable(SIM_TIMESTEP_MS)
    compass.enable(SIM_TIMESTEP_MS)

    trajectories = {
        name: make_trajectory(name, trajectory_rng) for name in TRAJECTORY_NAMES
    }

    features = []
    targets = []

    rollouts_per_trajectory = N_START_POINTS * N_PERTURBATIONS
    expected_samples = (
        len(TRAJECTORY_NAMES)
        * rollouts_per_trajectory
        * ROLLOUT_STEPS
    )

    print(
        f"[Collector] Starting data collection: {len(TRAJECTORY_NAMES)} trajectories, "
        f"{N_START_POINTS} start phases, {N_PERTURBATIONS} perturbations per phase, "
        f"{ROLLOUT_STEPS} steps per rollout, {expected_samples} expected samples."
    )

    for trajectory_index, (trajectory_name, trajectory) in enumerate(
        trajectories.items(), start=1
    ):
        n = len(trajectory["t"])
        if n < ROLLOUT_STEPS:
            raise ValueError(
                f"Trajectory {trajectory_name} has only {n} samples, "
                f"but ROLLOUT_STEPS={ROLLOUT_STEPS}."
            )

        # Avoid wrap-around discontinuities by ensuring every rollout stays
        # inside the stored reference trajectory.
        max_start = n - ROLLOUT_STEPS
        start_indices = np.linspace(
            0,
            max_start,
            N_START_POINTS,
            dtype=int,
        )

        print(
            f"[Collector] Trajectory {trajectory_index}/{len(TRAJECTORY_NAMES)}: "
            f"{trajectory_name}"
        )

        rollout_counter = 0
        for start_number, k0 in enumerate(start_indices, start=1):
            for perturb_number, (ex0, ey0, eth0) in enumerate(
                perturbations, start=1
            ):
                rollout_counter += 1

                x, y, theta = pose_from_tracking_error(
                    trajectory,
                    int(k0),
                    float(ex0),
                    float(ey0),
                    float(eth0),
                )

                # Remove any physical motion left from the previous rollout.
                right_motor.setVelocity(0.0)
                left_motor.setVelocity(0.0)
                translation.setSFVec3f([x, y, z0])
                rotation.setSFRotation([0.0, 0.0, 1.0, theta % (2.0 * np.pi)])
                robot_node.resetPhysics()

                for step in range(ROLLOUT_STEPS):
                    k = int(k0) + step

                    if step > 0:
                        x, y, theta = read_pose(gps, compass)

                    features.append(
                        tracking_features(x, y, theta, trajectory, k)
                    )
                    wr, wl = kanayama_controller(x, y, theta, trajectory, k)
                    targets.append(np.array([wr, wl], dtype=np.float32))

                    right_motor.setVelocity(wr)
                    left_motor.setVelocity(wl)

                    if not step_control_period(robot):
                        raise RuntimeError(
                            "Webots simulation stopped during data collection"
                        )

                if rollout_counter % 64 == 0 or (
                    start_number == N_START_POINTS
                    and perturb_number == N_PERTURBATIONS
                ):
                    print(
                        f"[Collector]   {trajectory_name}: rollout "
                        f"{rollout_counter}/{rollouts_per_trajectory} complete; "
                        f"samples collected: {len(features)}"
                    )

    print("[Collector] Data collection complete. Saving dataset...")
    right_motor.setVelocity(0.0)
    left_motor.setVelocity(0.0)

    features = np.asarray(features, dtype=np.float32).reshape(-1, 5)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 2)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET_PATH,
        features=features,
        targets=targets,
        dt=CONTROL_DT,
        wheel_radius=WHEEL_RADIUS,
        wheelbase=WHEELBASE,
        wheel_speed_limit=WHEEL_SPEED_LIMIT,
        kx=KX,
        ky=KY,
        kth=KTH,
        feature_names=np.array(
            ["e_x", "e_y", "e_theta", "v_ref", "omega_ref"]
        ),
        n_start_points=N_START_POINTS,
        n_perturbations=N_PERTURBATIONS,
        rollout_steps=ROLLOUT_STEPS,
        trajectory_seed=TRAJECTORY_SEED,
        perturbation_seed=PERTURBATION_SEED,
    )
    print(f"[Collector] Saved {len(features)} samples to {DATASET_PATH}")


if __name__ == "__main__":
    main()
