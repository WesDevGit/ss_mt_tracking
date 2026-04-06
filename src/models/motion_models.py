import numpy as np

dt = 1.5

F = np.array([
    [1.0, 0.0, dt,  0.0],
    [0.0, 1.0, 0.0, dt ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)


# constant turn f(x)

def f_ct(state_vector, dt):
    px, py, vx, vy, omega = state_vector.flatten()
    F_ct = np.array([[1, 0, np.sin(omega*dt)/omega    , -(1-np.cos(omega*dt))/omega, 0],
                [0, 1, (1-np.cos(omega*dt))/omega,  np.sin(omega*dt)          , 0],
                [0, 0, np.cos(omega*dt),           -np.sin(omega*dt),           0],
                [0, 0, np.sin(omega*dt),           np.cos(omega*dt),            0],
                [0, 0,                0,                          0,            1]])
    return F_ct

def f_ct_jaccob(state_vector, dt):
    px, py, vx, vy, omega = state_vector.flatten()
    phi = omega * dt
    s = np.sin(phi)
    c = np.cos(phi)

    A = s / omega
    B = (1.0 - c) / omega

    A_omega = (omega * dt * c - s) / (omega**2)
    B_omega = (omega * dt * s - (1.0 - c)) / (omega**2)

    J_f = np.array([
        [1.0, 0.0, A,   -B,  A_omega * vx - B_omega * vy],
        [0.0, 1.0, B,    A,  B_omega * vx + A_omega * vy],
        [0.0, 0.0, c,   -s, -dt * s * vx - dt * c * vy],
        [0.0, 0.0, s,    c,  dt * c * vx - dt * s * vy],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    return J_f