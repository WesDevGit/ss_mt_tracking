import numpy as np

def ss_baseline_seed():
    return np.random.seed(24)

def measurement_noise(R):
    return np.random.multivariate_normal(
        mean=[0.0, 0.0], cov=R
    ).reshape(2, 1)

dt = 1.5
sensor_position = np.array([[50.0], [50.0]], dtype=float)

# Very small process noise for near-deterministic validation
sigma_ax = 0.2
sigma_ay = 0.2
sigma_omega = 0.04 # rad/s

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
sigma_r = 1.5
sigma_theta = np.deg2rad(0.02)

R = np.diag([
    sigma_r**2,
    sigma_theta**2
]).astype(float)

# Very small initial covariance
P0 = np.diag([
    6.0,
    6.1,
    0.2,
    0.2,
    0.04
    
]).astype(float)
def smooth_transition(k, k0, k1, w0, w1):
    """
    Smoothly transition omega from w0 to w1 over k in [k0, k1]
    using a cosine blend.
    """
    if k <= k0:
        return w0
    if k >= k1:
        return w1

    tau = (k - k0) / (k1 - k0)
    blend = 0.5 * (1 - np.cos(np.pi * tau))
    return w0 + (w1 - w0) * blend

def omega_profile_track1(k):
    # 0–19: straight
    if k < 20:
        return 0.0

    # 20–30: smooth entry into left turn
    elif k < 30:
        return smooth_transition(k, 20, 30, 0.0, 0.035)

    # 30–50: hold moderate left turn
    elif k < 50:
        return 0.035

    # 50–60: smooth rollout to straight
    elif k < 60:
        return smooth_transition(k, 50, 60, 0.035, 0.0)

    # 60–75: straight
    elif k < 75:
        return 0.0

    # 75–85: smooth entry into sharper right turn
    elif k < 85:
        return smooth_transition(k, 75, 85, 0.0, -0.045)

    # 85–105: hold right turn
    elif k < 105:
        return -0.045

    # 105–115: smooth rollout back to straight
    elif k < 115:
        return smooth_transition(k, 105, 115, -0.045, 0.0)

    # 115–140: small oscillatory weave / S-turn
    elif k < 140:
        return 0.015 * np.sin(2 * np.pi * (k - 115) / 25.0)

    # 140+: straight
    else:
        return 0.0


truth_data = [
    {
        "id": 1,
        "x": np.array([
            [80.0],
            [40.0],
            [-1.0],
            [0.5],
            [0.0]
        ], dtype=float),
        "P": P0.copy(),
        "omega_profile": omega_profile_track1
    },
    {
        "id": 2,
        "x": np.array([[120.0],
                       [0.0],
                       [-1.0],
                       [0.5],
                       [-0.015]], dtype=float),
        "P": P0.copy(),
        "omega_profile":  lambda k: 0.0 if k < 20 else (-0.03 if k < 40 else (0.02 if k < 100 else (0.05 if k < 140 else 0.0)))
    },
    {
        "id": 4,
        "x": np.array([[80.0],
                       [-40.0],
                       [-1.0],
                       [0.5],
                       [-0.015]], dtype=float),
        "P": P0.copy(),
        "omega_profile":  lambda k: 0.0 if k < 40 else (-0.02 if k < 60 else (0.03 if k < 90 else (0.04 if k < 120 else 0.0)))
    },
    {
        "id": 3,
        "x": np.array([[20.0],
                       [70.0],
                       [-1.0],
                       [-1.0],
                       [0.02]], dtype=float),
        "P": P0.copy(),
        "omega_profile": lambda k: 0.04 if k < 15 else (0.0 if k < 35 else 0.02)
    }
]
# truth_data = [
#     {
#         "id": 1,
#         "x": np.array([[80.0],
#                        [40.0],
#                        [-1.0],
#                        [0.5],
#                        [0.0]], dtype=float),
#         "P": P0.copy(),
#         "omega_profile": lambda k: 0.0 if k < 20 else (0.03 if k < 40 else (-0.02 if k < 100 else 0.0))
#     },
#     # {
#     #     "id": 2,
#     #     "x": np.array([[120.0],
#     #                    [0.0],
#     #                    [1.0],
#     #                    [-1.0],
#     #                    [-0.015]], dtype=float),
#     #     "P": P0.copy(),
#     #     "omega_profile": lambda k: -0.015
#     # },
#     {
#         "id": 3,
#         "x": np.array([[20.0],
#                        [80.0],
#                        [-1.0],
#                        [-1.0],
#                        [0.02]], dtype=float),
#         "P": P0.copy(),
#         "omega_profile": lambda k: 0.02 if k < 15 else (0.0 if k < 35 else -0.02)
#     }
# ]

# No misses for validation
id_miss_index = {1: [1, 10,12,14,26,50,55,60], 2: [25,26,27,28, 50,51,52, 75]}