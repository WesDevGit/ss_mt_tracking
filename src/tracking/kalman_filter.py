import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x_hat_km1_km1, P_km1_km1):
        self.F = F                  # state transition matrix (may be dynamic depending on the model?)
        self.H = H                  # measurement matrix (may be dynamic depending on the model?)
        self.Q = Q                  # process noise covariance (may be dynamic?)
        self.R = R                  # measurement noise covariance (may be dynamic?)
                                    # we may also have other noise parameters here such as process noise, control input, etc.

        self.x_hat_km1_km1 = x_hat_km1_km1  # prior predicated state estimate x_hat_{k-1|k-1}  
        self.P_km1_km1 = P_km1_km1 # prior predicated covariance P_{k-1|k-1}   
        self.x_hat_k_km1 = None  # posterior state estimate x_hat_{k|k-1}
        self.P_k_km1 = None  # posterior predicated covariance P_{k|k-1}
        self.x_hat_k_k = None # updated state estimate x_hat_{k|k}
        self.P_k_k = None  #  updated covariance P_{k|k}
        
    def predict(self):
        """
        Prediction step:
            x_hat_{k|k-1} = F x_hat_{k-1|k-1}
            P_{k|k-1} = F P_{k-1|k-1} F^T + Q
        """
        self.x_hat_k_km1 = self.F @ self.x_hat_km1_km1   # state equation. F is a map from previous state to the new predicted state and we are multiplying this
        # matrix by our previous state estimate to give us our new predicted state estimate. If available we add in process noise or control input.

        self.P_k_km1 = self.F @ self.P_km1_km1 @ self.F.T + self.Q   # Predict the error covariance forward one step. (P_{k|k-1})
        # F maps the previous state uncertainty through the motion model,
        # and F.T completes the covariance transformation.
        # Q adds new uncertainty from process noise.

        return self.x_hat_k_km1, self.P_k_km1   # Returns the predicted state estimate x_hat_{k|k-1}
        # and predicted error covariance P_{k|k-1},
        # both prior to incorporating the measurement (z_k) at time k.

    def update(self, z_k):
        """
        Update step:
            y_k = z_k - H x_hat_{k|k-1}
            S_k = H P_{k|k-1} H^T + R
            K_k = P_{k|k-1} H^T S_k^{-1}
            x_hat_{k|k} = x_hat_{k|k-1} + K_k y_k
            P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T
        """
        # we receive a measurement so now we want to update our prior prediction.
        self.y_k = z_k - (self.H @ self.x_hat_k_km1)   # residual (how far is our measurement from our predicted measurement?
        # [H @ x_hat_{k|k-1} maps our prediction into measurement space.])

        self.S_k = self.H @ self.P_k_km1 @ self.H.T + self.R   # residual uncertainty (how certain are we about our residual?)
        # H maps P_{k|k-1} into measurement space, H.T completes the covariance transformation,
        # and we add assumed measurement noise covariance.

        self.K_k = self.P_k_km1 @ self.H.T @ np.linalg.solve(
            self.S_k, np.eye(self.S_k.shape[0])
        )   # Kalman Gain, how much the filter should correct the prediction using the measurement.
        # Here we create a matrix that maps measurement error into state correction.

        self.x_hat_k_k = self.x_hat_k_km1 + self.K_k @ self.y_k   # corrects predicted state using measurement.
        # Move estimate toward measurement but only as much as the gain says.
        # We multiply our gain by our innovation which tells us how much to correct our estimate.
        # Large K tends to mean prediction is uncertain and measurement is more accurate so update
        # x_hat_{k|k} to be closer to the measurement.
        # Small K tends to mean prediction is very certain and measurement is noisy so move more toward x_hat_{k|k-1}.

        I = np.eye(self.P_k_km1.shape[0])
        self.P_k_k = (
            (I - self.K_k @ self.H)
            @ self.P_k_km1
            @ (I - self.K_k @ self.H).T
            + self.K_k @ self.R @ self.K_k.T
        )   # Update the predicted state covariance P_{k|k-1} to the corrected covariance P_{k|k}.
        # The first term transforms the predicted covariance through the measurement update.
        # The second term adds the contribution of measurement noise, mapped from measurement space into state space by K.
        self.x_hat_km1_km1 = self.x_hat_k_k # Reset km1|km1 so next time you predict all it will used model with next update.
        self.P_km1_km1 = self.P_k_k
        return self.x_hat_k_k, self.P_k_k, self.y_k, self.S_k, self.K_k

    def step(self, z_k):
        self.predict()
        return self.update(z_k)