import numpy as np

def ss_baseline_seed():
    return np.random.seed(3)

def measurement_noise(R):
    return np.random.multivariate_normal(
        mean=[0.0, 0.0], cov=R
    ).reshape(2, 1)

dt = 1.5
sensor_position = np.array([[50.0], [50.0]], dtype=float)

# Very small process noise for near-deterministic validation
sigma_ax = 0.1
sigma_ay = 0.1
sigma_omega = 0.02 # rad/s

G = np.array([
    [0.5 * dt**2, 0.0,          0.0],
    [0.0,         0.5 * dt**2,  0.0],
    [dt,          0.0,          0.0],
    [0.0,         dt,           0.0],
    [0.0,         0.0,          dt],
], dtype=float)

Sigma_w = np.diag([
    sigma_ax**2,
    sigma_ay**2,
    sigma_omega**2
]).astype(float)

Q = G @ Sigma_w @ G.T

# Very small measurement noise
sigma_r = 50.5
sigma_theta = np.deg2rad(0.03)

R = np.diag([
    sigma_r**2,
    sigma_theta**2
]).astype(float)

# Very small initial covariance
P0 = np.diag([
    4.0,
    4.1,
    0.3,
    0.2,
    0.04
]).astype(float)

truth_data = [
    {
        "id": 1,
        "x": np.array([[80.0],
                       [40.0],
                       [-1.0],
                       [0.5],
                       [0.0]], dtype=float),
        "P": P0.copy(),
        "omega_profile": lambda k: 0.0 if k < 20 else (0.03 if k < 40 else (-0.02 if k < 100 else 0.0))
    },
    # {
    #     "id": 2,
    #     "x": np.array([[400.0],
    #                    [0.0],
    #                    [1.0],
    #                    [-1.0],
    #                    [-0.015]], dtype=float),
    #     "P": P0.copy(),
    #     "omega_profile": lambda k: -0.015
    # },
    {
        "id": 3,
        "x": np.array([[20.0],
                       [80.0],
                       [-1.0],
                       [-1.0],
                       [0.02]], dtype=float),
        "P": P0.copy(),
        "omega_profile": lambda k: 0.02 if k < 15 else (0.0 if k < 35 else -0.02)
    }
]

# No misses for validation
id_miss_index = {1: [1, 10,12,14,26,50,55,60]}