import numpy as np

class ExtendedKalmanFilter:
    def __init__(
        self,
        Q,
        R,
        x_hat_km1_km1,
        P_km1_km1,
        sensor_position,
        dt=1.5,
        omega_eps=1e-6,
    ):
        self.Q = Q
        self.R = R

        self.x_hat_km1_km1 = x_hat_km1_km1.copy()
        self.P_km1_km1 = P_km1_km1.copy()

        self.x_hat_k_km1 = None
        self.P_k_km1 = None
        self.x_hat_k_k = None
        self.P_k_k = None
        self.sensor_position = sensor_position.copy()

        self.dt = dt
        self.omega_eps = omega_eps

        self.F = None
        self.H = None
        self.h = None
        self.y_k = None
        self.S_k = None
        self.K_k = None

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def normalize_innovation(self, y):
        y = y.copy()
        y[1, 0] = self.normalize_angle(y[1, 0])
        return y
    
    def f_ct(self, x):
        px, py, vx, vy, omega = x.flatten()
        dt = self.dt

        if abs(omega) < self.omega_eps:
            return np.array([
                [px + dt * vx - 0.5 * dt**2 * omega * vy],
                [py + dt * vy + 0.5 * dt**2 * omega * vx],
                [vx - dt * omega * vy],
                [vy + dt * omega * vx],
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
        

    def f_ct_jacobian(self, x):
        px, py, vx, vy, omega = x.flatten()
        dt = self.dt

        if abs(omega) < self.omega_eps:
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

    def h_range_az(self, x):
        px, py, vx, vy, omega = x.flatten()
        sx, sy = self.sensor_position.flatten()

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
            [dx / r,    dy / r,    0.0, 0.0, 0.0],
            [-dy / r2,  dx / r2,   0.0, 0.0, 0.0],
        ], dtype=float)

        return h_x, H

    def predicted_measurement(self):
        h, H = self.h_range_az(self.x_hat_k_km1)
        return h, H

    def gating_stats(self, z_k):
        z_hat, H = self.predicted_measurement()
        y = self.normalize_innovation(z_k - z_hat)
        S = H @ self.P_k_km1 @ H.T + self.R
        d2 = float(y.T @ np.linalg.solve(S, y))
        return z_hat, H, y, S, d2

    def predict(self):
        self.x_hat_k_km1 = self.f_ct(self.x_hat_km1_km1)
        self.F = self.f_ct_jacobian(self.x_hat_km1_km1)
        self.P_k_km1 = self.F @ self.P_km1_km1 @ self.F.T + self.Q
        self.h, self.H = self.predicted_measurement()
        return self.x_hat_k_km1, self.P_k_km1

    def update(self, z_k):
        self.h, self.H, self.y_k, self.S_k, _ = self.gating_stats(z_k)

        self.K_k = self.P_k_km1 @ self.H.T @ np.linalg.solve(
            self.S_k, np.eye(self.S_k.shape[0])
        )

        self.x_hat_k_k = self.x_hat_k_km1 + self.K_k @ self.y_k

        I = np.eye(self.P_k_km1.shape[0])
        self.P_k_k = (
            (I - self.K_k @ self.H)
            @ self.P_k_km1
            @ (I - self.K_k @ self.H).T
            + self.K_k @ self.R @ self.K_k.T
        )

        self.x_hat_km1_km1 = self.x_hat_k_k
        self.P_km1_km1 = self.P_k_k
        return self.x_hat_k_k, self.P_k_k, self.y_k, self.S_k, self.K_k

    def step(self, z_k):
        self.predict()
        return self.update(z_k)