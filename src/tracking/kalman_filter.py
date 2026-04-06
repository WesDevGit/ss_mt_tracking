import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x_hat_km1_km1, P_km1_km1):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        self.x_hat_km1_km1 = x_hat_km1_km1
        self.P_km1_km1 = P_km1_km1

        self.x_hat_k_km1 = None
        self.P_k_km1 = None
        self.x_hat_k_k = None
        self.P_k_k = None

        self.y_k = None
        self.S_k = None
        self.K_k = None

    def predict(self):
        self.x_hat_k_km1 = self.F @ self.x_hat_km1_km1
        self.P_k_km1 = self.F @ self.P_km1_km1 @ self.F.T + self.Q
        return self.x_hat_k_km1, self.P_k_km1

    def predicted_measurement(self):
        z_hat = self.H @ self.x_hat_k_km1
        H = self.H
        return z_hat, H

    def normalize_innovation(self, y):
        return y

    def gating_stats(self, z_k):
        z_hat, H = self.predicted_measurement()
        y = self.normalize_innovation(z_k - z_hat)
        S = H @ self.P_k_km1 @ H.T + self.R
        d2 = float(y.T @ np.linalg.solve(S, y))
        return z_hat, H, y, S, d2

    def update(self, z_k):
        z_hat, H = self.predicted_measurement()
        self.y_k = self.normalize_innovation(z_k - z_hat)
        self.S_k = H @ self.P_k_km1 @ H.T + self.R

        self.K_k = self.P_k_km1 @ H.T @ np.linalg.solve(
            self.S_k, np.eye(self.S_k.shape[0])
        )

        self.x_hat_k_k = self.x_hat_k_km1 + self.K_k @ self.y_k

        I = np.eye(self.P_k_km1.shape[0])
        self.P_k_k = (
            (I - self.K_k @ H)
            @ self.P_k_km1
            @ (I - self.K_k @ H).T
            + self.K_k @ self.R @ self.K_k.T
        )

        self.x_hat_km1_km1 = self.x_hat_k_k
        self.P_km1_km1 = self.P_k_k
        return self.x_hat_k_k, self.P_k_k, self.y_k, self.S_k, self.K_k

    def step(self, z_k):
        self.predict()
        return self.update(z_k)