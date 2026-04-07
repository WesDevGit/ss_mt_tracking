
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.track import Track

class TrackManager:
    def __init__(self, tracks, gate_threshold, track_factory=None):
        self.tracks = tracks
        self.gate_threshold = gate_threshold
        self.track_factory = track_factory
        self.next_track_id = max((t.track_id for t in tracks), default=0) + 1
        self.scan_log = {}
        self.pred_log = {}
        self.gate_log = {}
        self.assoc_log = {}
        self.consistency_log = {}
        self.update_log = {}
        
    def get_new_track_id(self):
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
                        
    def log_error(self, track, truth):
        try:
            epsilon = (track.kf.x_hat_k_k - truth).T @ np.linalg.solve(track.kf.P_k_k, (track.kf.x_hat_k_k - truth))
        except Exception as e:
            epsilon = (track.kf.x_hat_k_k - truth).T @ np.linalg.pinv(track.kf.P_k_k) @ (track.kf.x_hat_k_k - truth)
        if track.track_id not in self.update_log.keys():
            self.update_log[track.track_id] = [epsilon.item()]
        else:
            self.update_log[track.track_id].append(epsilon.item())
            
    def tentative_track(self, track):
        "Add a tentative track to the track list and initialize its current fields."
        track.kf.x_hat_k_km1 = track.kf.x_hat_km1_km1.copy()
        track.kf.x_hat_k_k = track.kf.x_hat_km1_km1.copy()
        track.kf.P_k_km1 = track.kf.P_km1_km1.copy()
        track.kf.P_k_k = track.kf.P_km1_km1.copy()

        self.tracks.append(track)
        return track.track_id
        
    def delete_track(self, track):
        self.tracks = [t for t in self.tracks if t.track_id != track.track_id]
        
    
    def predict_all(self):
        "Predict all tracks x_hat_k|km1 and P_k|km1"
        for track in self.tracks:
            track.predict()
            
    def create_tentative_from_measurement(self, z_k):
        if self.track_factory is None:
            return None

        track_id = self.get_new_track_id()
        track = self.track_factory(track_id, z_k)
        return self.tentative_track(track)
    
    def mahalanobis_distance(self, track, z_k):
        """
        Point-to-distribution hypothesis testing.
        """
        _, _, y, S, d2 = track.kf.gating_stats(z_k)
        return y, S, d2

    def build_cost_matrix(self, measurements):
        """Cost Matrix NxM where N are tracks and M are measurements. Elements of the cost matrix
        are the result of the mahalanobis distance for a give track_i ,measurement_j pair.
        
        * P(d^2 =< gate_threshold | H_0) = alpha -> cost_matix[i,j] = d2
        * P(d^2 > gate_threshold | H_0) = 1- alpha -> cost_matrix[i,j] = 100000 (set to invalid cost assignment value)
        """
        N = len(self.tracks)
        M = len(measurements)

        invalid_cost = 1e6
        cost_matrix = np.full((N, M), invalid_cost, dtype=float)

        gate_info = {}

        for i, track in enumerate(self.tracks):
            track_id = track.track_id
            gate_info[track_id] = {}

            for j, z in enumerate(measurements):
                y, S, d2 = self.mahalanobis_distance(track, z)

                passed_gate = d2 <= self.gate_threshold

                gate_info[track_id][j + 1] = {
                    "measurement": z,
                    "y": y,
                    "S": S,
                    "d2": d2,
                    "passed_gate": passed_gate
                }

                if passed_gate:
                    cost_matrix[i, j] = d2

        return cost_matrix, gate_info

    def gnn_associate(self, measurements, k):
        """
        Build cost matrix and solve the linear solve assignment problem.
        
        The number of assignments in a nxm matrix where n neq m is min(n,m)
        Check to see is the assignment for a track is within gate threshold
        If so we assign tracks to measurements, mark unassigned tracks, and mark unassigned measurements.
        """
        cost_matrix, gate_info = self.build_cost_matrix(measurements)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        assigned_tracks = set()
        assigned_measurements = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.gate_threshold:
                track_id = self.tracks[i].track_id
                if track_id not in self.consistency_log.keys():
                    self.consistency_log[track_id] = [gate_info[track_id][j + 1]["d2"]]
                else:
                    self.consistency_log[track_id].append(gate_info[track_id][j + 1]["d2"])
                assignments.append({
                    "track_id": track_id,
                    "measurement_number": j + 1,
                    "measurement": measurements[j],
                    "y": gate_info[track_id][j + 1]["y"],
                    "S": gate_info[track_id][j + 1]["S"],
                    "d2": gate_info[track_id][j + 1]["d2"],
                })

                assigned_tracks.add(i)
                assigned_measurements.add(j)
        if k not in self.gate_log:
            self.gate_log[k] = {}

        for assign in assignments:
            track_id = assign["track_id"]
            meas_num = assign["measurement_number"]

            if track_id not in self.gate_log[k]:
                self.gate_log[k][track_id] = {}

            self.gate_log[k][track_id][meas_num] = {
                "y": assign["y"],
                "S": assign["S"],
                "d2": assign["d2"]
            }
        unassigned_tracks = [{"track_id": self.tracks[i].track_id}
                            for i in range(len(self.tracks))
                            if i not in assigned_tracks]
        unassigned_measurements = [{"measurement_number":j+1, "measurement": measurements[j]} for j in range(len(measurements)) if j not in assigned_measurements]
        self.assoc_log[k] = {'cost_matrix': cost_matrix,"assignments":assignments, "unassigned_tracks": unassigned_tracks, "unassigned_measurements": unassigned_measurements}
        return assignments, unassigned_tracks, unassigned_measurements, cost_matrix