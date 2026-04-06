# import numpy as np
# # This is the first project where I am building out my system. Here we implement a seed to begin validating everything is working.
# def ss_baseline_seed():
#     return np.random.seed(3)

# def measurement_noise(R):
#     return np.random.multivariate_normal(
#         mean=[0.0, 0.0], cov=R
#     ).reshape(2, 1)

# dt = 1.5
# sensor_position = np.array([[0.0], [0.0]], dtype=float)

# # Process-noise standard deviations
# sigma_ax = 4.0
# sigma_ay = 5.4
# sigma_omega = np.deg2rad(0.5)   # rad/s process noise on turn rate

# G = np.array([
#     [0.5 * dt**2, 0.0,          0.0],
#     [0.0,         0.5 * dt**2,  0.0],
#     [dt,          0.0,          0.0],
#     [0.0,         dt,           0.0],
#     [0.0,         0.0,          dt],
# ], dtype=float)

# Sigma_w = np.diag([
#     sigma_ax**2,
#     sigma_ay**2,
#     sigma_omega**2
# ]).astype(float)

# Q = G @ Sigma_w @ G.T

# # Measurement-noise standard deviations
# sigma_r = 10.8
# sigma_theta = np.deg2rad(0.02)

# R = np.diag([
#     sigma_r**2,
#     sigma_theta**2
# ]).astype(float)

# # Initial track covariance
# P0 = np.diag([
#     30.0,
#     35.0,
#     35.0,
#     35.0,
#     0.05**2
# ]).astype(float)

# truth_data = [ 
#     {
#         "id": 1,
#         "x": np.array([[80.0],
#                        [0.0],
#                        [-1.0],
#                        [0.5],
#                        [0.01]], dtype=float),
#         "P": P0
#     },
#     # {
#     #     "id": 2,
#     #     "x": np.array([[-40.0],
#     #                    [-10.0],
#     #                    [1.0],
#     #                    [-1.0],
#     #                    [0.02]], dtype=float),
#     #     "P": P0
#     # },
#     # {
#     #     "id": 3,
#     #     "x": np.array([[20.0],
#     #                    [1.0],
#     #                    [-1.0],
#     #                    [-1.0],
#     #                    [0.01]], dtype=float),
#     #     "P": P0
#     # }
# ]
id_miss_index = {}

import numpy as np

def ss_baseline_seed():
    return np.random.seed(3)

def measurement_noise(R):
    return np.random.multivariate_normal(
        mean=[0.0, 0.0], cov=R
    ).reshape(2, 1)

dt = 1.5
sensor_position = np.array([[0.0], [0.0]], dtype=float)

# Very small process noise for near-deterministic validation
sigma_ax = 0.01
sigma_ay = 0.01
sigma_omega = 1e-4   # rad/s

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
sigma_r = 0.05
sigma_theta = np.deg2rad(0.01)

R = np.diag([
    sigma_r**2,
    sigma_theta**2
]).astype(float)

# Very small initial covariance
P0 = np.diag([
    5.0,
    4.01,
    3.0,
    2.0,
    1e-2
]).astype(float)

truth_data = [
    {
        "id": 1,
        "x": np.array([
            [80.0],
            [0.0],
            [-1.0],
            [0.5],
            [0.01]
        ], dtype=float),
        "P": P0.copy()
    }
]

# No misses for validation
id_miss_index = {}