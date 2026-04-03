import numpy as np
# This is the first project where I am building out my system. Here we implement a seed to begin validating everything is working.

np.random.seed(1)

def measurement_noise(R):
    return np.random.multivariate_normal(mean=[0, 0], cov=R).reshape(2, 1)
sigma_ax = 1.2
sigma_ay = 0.8
dt = 1.5
G = np.array([
    [0.5 * dt**2, 0.0],
    [0.0, 0.5 * dt**2],
    [dt, 0.0],
    [0.0, dt]
], dtype=float)

Sigma_a = np.array([
    [sigma_ax**2, 0.0],
    [0.0, sigma_ay**2]
], dtype=float)

Q = G @ Sigma_a @ G.T

sigma_x = 4.0
sigma_y = 6.0
rho = 0.25

R = np.array([
    [sigma_x**2,           rho * sigma_x * sigma_y],
    [rho * sigma_x * sigma_y, sigma_y**2]
], dtype=float)
P0 = np.diag([R[0,0], R[1,1], 100.0, 100.0])


truth_data = [ 
                {
        "id": 1,
        "x": np.array([[80.0],
                       [111.0],
                       [  3.0],
                       [ 0.0]], dtype=float),
        "P": P0
    },
    {
        "id": 2,
        "x": np.array([[25.0],
                       [111.0],
                       [  1.0],
                       [ -1.0]], dtype=float),
        "P": P0
    },
    {
        "id": 3,
        "x": np.array([[10.0],
                       [80.0],
                       [  1.0],
                       [ -1.0]], dtype=float),
        "P": P0
    },
    {
        "id": 4,
        "x": np.array([[20.0],
                       [140.0],
                       [  1.0],
                       [ 1.0]], dtype=float),
        "P": P0
    },
    {
        "id": 5,
        "x": np.array([[80.0],
                       [140.0],
                       [  -1.0],
                       [ 1.0]], dtype=float),
        "P": P0
    }
]
id_miss_index = {1: [10, 12, 13, 14, 15, 25]}