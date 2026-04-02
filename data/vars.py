import numpy as np

dt = 1.5

F = np.array([
    [1.0, 0.0, dt,  0.0],
    [0.0, 1.0, 0.0, dt ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)

H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
], dtype=float)

sigma_ax = 1.2
sigma_ay = 0.8

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

tracks = [
    {
        "id": 1,
        "x": np.array([[ 92.0],
                       [210.0],
                       [  4.0],
                       [ -2.0]], dtype=float),
        "P": np.array([
            [20.0, 0.0, 4.0, 0.0],
            [0.0, 18.0, 0.0, 3.0],
            [4.0, 0.0, 9.0, 0.0],
            [0.0, 3.0, 0.0, 6.0]
        ], dtype=float)
    },
    {
        "id": 2,
        "x": np.array([[154.0],
                       [ 78.0],
                       [ -3.0],
                       [  6.0]], dtype=float),
        "P": np.array([
            [25.0, 0.0, 6.0, 0.0],
            [0.0, 16.0, 0.0, 4.0],
            [6.0, 0.0, 9.0, 0.0],
            [0.0, 4.0, 0.0, 4.0]
        ], dtype=float)
    },
    {
        "id": 3,
        "x": np.array([[166.0],
                       [ 96.0],
                       [ -5.0],
                       [ -2.0]], dtype=float),
        "P": np.array([
            [18.0, 0.0, 3.0, 0.0],
            [0.0, 25.0, 0.0, 6.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 6.0, 0.0, 9.0]
        ], dtype=float)
    },
    {
        "id": 4,
        "x": np.array([[241.0],
                       [224.0],
                       [  6.0],
                       [ -3.0]], dtype=float),
        "P": np.array([
            [16.0, 0.0, 4.0, 0.0],
            [0.0, 20.0, 0.0, 5.0],
            [4.0, 0.0, 9.0, 0.0],
            [0.0, 5.0, 0.0, 9.0]
        ], dtype=float)
    }
]

measurements = [
    np.array([[150.7], [ 89.1]], dtype=float),
    np.array([[158.0], [ 91.8]], dtype=float),
    np.array([[249.6], [217.3]], dtype=float),
    np.array([[166.8], [103.2]], dtype=float),
    np.array([[121.5], [228.4]], dtype=float)
]