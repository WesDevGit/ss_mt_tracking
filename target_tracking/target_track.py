import numpy as np
from scipy.optimize import linear_sum_assignment
from data import vars
from target_tracking.kalman_filter import KalmanFilter

class Track:

    def __init__(self, track_id, kf, tentative = False):
        self.track_id = track_id 
        self.kf = kf # filter object
        self.missed_count = 0 # number of missed detections
        self.age = 1 # track age (number of times track_id remains alive)
        self.hit_count = 1 
        self.tentative = tentative # determines if the track is tentative or active
        self.track_predicted_states = {}

    def predict(self):
        "Used to predict single track but do not coast"
        return self.kf.predict()

    def update(self, z_k):
        "Update track, its age, set missed count to 0 and use measurement"
        self.age += 1
        self.missed_count = 0
        return self.kf.update(z_k)
    
    def coast(self):
        "coast current track prediction forward into current state estimate"
        self.kf.x_hat_km1_km1 = self.kf.x_hat_k_km1
        self.kf.P_km1_km1 = self.kf.P_k_km1
        self.kf.x_hat_k_k = self.kf.x_hat_k_km1
        self.kf.P_k_k = self.kf.P_k_km1 # Here we never intend to update so we use our coast our predication forward.

    def miss(self):
        "Indicates track has missed detection. increases age of track and missed count."
        self.age += 1
        self.missed_count += 1
        
    def promote_track(self):
        self.tentative = False
        
        
            
class TrackManager:
    def __init__(self, tracks, gate_threshold):
        self.tracks = tracks # list of track objects
        self.gate_threshold = gate_threshold # currently based on Mahalanobis distance (chi squared) with n
                                             # degrees of freedom (measurement vector dim) and confidence
        self.next_track_id = max((t.track_id for t in tracks), default=0) + 1
        self.scan_log = {}
        self.pred_log = {}
        self.gate_log = {}
        self.assoc_log = {}
        
    def get_new_track_id(self):
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
                        
    def tentative_track(self, z_k, track_id):
        "Build a tentative track add it to the track list it now has age 1 missed count 0 and tentative set to True"
        t_track = {
            "id": track_id,
            "x": np.array([[z_k[0].item()],
                        [z_k[1].item()],
                        [ 0.0],
                        [0.0]], dtype=float),
            "P": np.array([
                [36.0, 0.0, 0.0,  0.0],
                [0.0, 36.0, 0.0,  0.0],
                [0.0, 0.0, 100.0, 0.0],
                [0.0, 0.0, 0.0, 100.0]
            ], dtype=float)
        }
        ten_track = Track(t_track['id'], KalmanFilter(vars.F, vars.H, vars.Q, vars.R, t_track['x'], t_track["P"]), True)
        ten_track.kf.x_hat_k_km1 = ten_track.kf.x_hat_km1_km1
        ten_track.kf.x_hat_k_k = ten_track.kf.x_hat_km1_km1
        ten_track.kf.P_k_km1 = ten_track.kf.P_km1_km1
        ten_track.kf.P_k_k = ten_track.kf.P_km1_km1
        
        self.tracks.append(ten_track)
        return track_id
        
    def delete_track(self, track):
        self.tracks = [t for t in self.tracks if t.track_id != track.track_id]
        
    
    def predict_all(self):
        "Predict all tracks x_hat_k|km1 and P_k|km1"
        for track in self.tracks:
            track.predict()

    def mahalanobis_distance(self, track, z_k):
        """
        Point to distribution hypothesis testing. Answers the question, for a given measurement, how confident am I
        this measurement belongs to this track?
        * H_0: Measurement came from this track
        * H_1: Measurement is clutter or from another target
        """
        kf = track.kf
        y = z_k - (kf.H @ kf.x_hat_k_km1)
        S = kf.H @ kf.P_k_km1 @ kf.H.T + kf.R
        d2 = y.T @ np.linalg.solve(S, y)
        return y, S, d2.item()

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