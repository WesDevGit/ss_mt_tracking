import numpy as np
from src.models import measurement_models
from src.models import motion_models

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def wrap_bearing_residual(y):
    y = y.copy()
    y[1, 0] = wrap_angle(y[1, 0])
    return y

def f_ct(state_vector, dt, omega_eps=1e-6):
    """
    Nonlinear coordinated-turn state transition.
    Input and output state ordering:
    [px, py, vx, vy, omega]^T
    """
    px, py, vx, vy, omega = state_vector.flatten()

    if abs(omega) < omega_eps:
        return np.array([
            [px + dt * vx],
            [py + dt * vy],
            [vx],
            [vy],
            [omega],
        ], dtype=float)

    phi = omega * dt
    s = np.sin(phi)
    c = np.cos(phi)

    A = s / omega
    B = (1.0 - c) / omega

    return np.array([
        [px + A * vx - B * vy],
        [py + B * vx + A * vy],
        [c * vx - s * vy],
        [s * vx + c * vy],
        [omega],
    ], dtype=float)

def f_ct_jaccob(state_vector, dt, omega_eps=1e-6):
    """
    Jacobian of coordinated-turn transition with respect to state.
    """
    px, py, vx, vy, omega = state_vector.flatten()

    if abs(omega) < omega_eps:
        return np.array([
            [1.0, 0.0, dt,  0.0, -0.5 * dt**2 * vy],
            [0.0, 1.0, 0.0, dt,   0.5 * dt**2 * vx],
            [0.0, 0.0, 1.0, 0.0, -dt * vy],
            [0.0, 0.0, 0.0, 1.0,  dt * vx],
            [0.0, 0.0, 0.0, 0.0,  1.0],
        ], dtype=float)

    phi = omega * dt
    s = np.sin(phi)
    c = np.cos(phi)

    A = s / omega
    B = (1.0 - c) / omega

    A_omega = (omega * dt * c - s) / (omega**2)
    B_omega = (omega * dt * s - (1.0 - c)) / (omega**2)

    return np.array([
        [1.0, 0.0, A,   -B,  A_omega * vx - B_omega * vy],
        [0.0, 1.0, B,    A,  B_omega * vx + A_omega * vy],
        [0.0, 0.0, c,   -s, -dt * s * vx - dt * c * vy],
        [0.0, 0.0, s,    c,  dt * c * vx - dt * s * vy],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

def h_range_az(state_vector, sensor_position):
    """
    Nonlinear range-bearing measurement model and its Jacobian.
    """
    px, py, vx, vy, omega = state_vector.flatten()
    sx, sy = sensor_position.flatten()

    dx = px - sx
    dy = py - sy
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)

    if r < 1e-9:
        raise ValueError("Target too close to sensor for stable range-bearing Jacobian.")

    h_x = np.array([
        [r],
        [np.arctan2(dy, dx)],
    ], dtype=float)

    H = np.array([
        [dx / r,      dy / r,      0.0, 0.0, 0.0],
        [-dy / r2,    dx / r2,     0.0, 0.0, 0.0],
    ], dtype=float)

    return h_x, H  

def generate_truth(
    n_steps,
    truth_data,
    id_miss_index,
    R,
    dt,
    measurement_noise,
    sensor_position,
):
    track_truths = []
    measure_data = []

    for state in truth_data:
        x_k = state["x"].copy()

        truth_entry = {
            "id": state["id"],
            "x_states": [x_k.copy()],
        }

        meas_entry = {
            "id": state["id"],
            "measurements": [],
        }

        omega_profile = state.get("omega_profile", None)

        for k in range(n_steps):
            if omega_profile is not None:
                x_k[4, 0] = omega_profile(k)

            x_k = f_ct(x_k, dt)
            truth_entry["x_states"].append(x_k.copy())

            h, H = h_range_az(x_k, sensor_position)
            z_k = h + measurement_noise(R)
            z_k[1, 0] = wrap_angle(z_k[1, 0])
            meas_entry["measurements"].append(z_k)

        track_truths.append(truth_entry)
        measure_data.append(meas_entry)

    truth_states = {}
    for tid in track_truths:
        truth_states[tid["id"]] = [x for x in tid["x_states"]]

    truth_positions = {}
    for tid in track_truths:
        truth_positions[tid["id"]] = [x[:2, :] for x in tid["x_states"]]

    truth_velocities = {}
    for tid in track_truths:
        truth_velocities[tid["id"]] = [x[2:4, :] for x in tid["x_states"]]

    truth_times = [i * dt for i in range(n_steps + 1)]

    scans = build_scans(measure_data, id_miss_index)
    truth_exists = truth_misses(track_truths, id_miss_index, len(truth_times))

    return truth_states, truth_positions, truth_velocities, truth_times, truth_exists, scans

### Truth data is a list of dictionaries with {"id":, "x", "P"}
 
def build_scans(measure_data, miss_indices):
    all_measurements = [md['measurements'] for md in measure_data]
    id_to_idx = {md['id']: i for i, md in enumerate(measure_data)}
    pos_miss = {id_to_idx[tid]: indices for tid, indices in miss_indices.items()
                if tid in id_to_idx}

    n_scans = len(all_measurements[0])
    scans = {}

    for i in range(n_scans):
        scan_index = i + 1
        scan_measurements = []
        for track_idx, track_meas in enumerate(all_measurements):
            if track_idx in pos_miss and scan_index in pos_miss[track_idx]:
                continue
            scan_measurements.append(track_meas[i])
        scans[scan_index] = scan_measurements
    return scans

def truth_misses(track_truths, id_miss_index, n_times):
    truth_exists = {}
    for tid in track_truths:
        truth_exists[tid['id']] = [1] * n_times
        if tid['id'] in id_miss_index:
            for idx in id_miss_index[tid['id']]:
                truth_exists[tid['id']][idx] = 0
    return truth_exists
