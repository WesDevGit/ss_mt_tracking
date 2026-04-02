import numpy as np
from scipy.optimize import linear_sum_assignment
from data import vars
from target_tracking.kalman_filter import KalmanFilter

class Track:

    def __init__(self, track_id, kf, tenative = False):
        self.track_id = track_id 
        self.kf = kf # filter object
        self.missed_count = 0 # number of missed detections
        self.age = 1 # track age (number of times track_id remains alive)
        self.hit_count = 1 
        self.tenative = tenative # determines if the track is tenative or active

    def predict(self):
        "Used to predict single track but do not propagate"
        return self.kf.predict()

    def update(self, z_k):
        "Update track, its age, set missed count to 0 and use measurement"
        self.age += 1
        self.missed_count = 0
        return self.kf.update(z_k)
    
    def propagate(self):
        "Propagate current track prediction forward into current state estimate"
        self.kf.x_hat_km1_km1 = self.kf.x_hat_k_km1
        self.kf.P_km1_km1 = self.kf.P_k_km1
        self.kf.x_hat_k_k = self.kf.x_hat_k_km1
        self.kf.P_k_k = self.kf.P_k_km1 # Here we never intend to update so we use our propagate our predication forward.

    def miss(self):
        "Indicates track has missed detection. increases age of track and missed count."
        self.age += 1
        self.missed_count += 1
        
    def promote_track(self):
        self.tenative = False
        
        
            
class TrackManager:
    def __init__(self, tracks, gate_threshold):
        self.tracks = tracks # list of track objects
        self.gate_threshold = gate_threshold # currently based on Mahalanobis distance (chi squared) with n
                                             # degrees of freedom (measurement vector dim) and confidence
        self.next_track_id = max((t.track_id for t in tracks), default=0) + 1
    
    def get_new_track_id(self):
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
                        
    def tenative_track(self, z_k, track_id):
        "Build a tenative track add it to the track list it now has age 1 missed count 0 and tenative set to True"
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
        del_id = track.track_id
        for i, trx in enumerate(self.tracks):
            if trx.track_id == del_id:
                self.tracks.remove(self.tracks[i])
        
    
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
        return d2.item()

    def build_cost_matrix(self, measurements):
        """Cost Matrix NxM where N are tracks and M are measurements. Elements of the cost matrix
        are the result of the mahalanobis distance for a give track_i ,measurement_j pair.
        
        * P(d^2 =< gate_threshold | H_0) = alpha -> cost_matix[i,j] = d2
        * P(d^2 > gate_threshold | H_0) = 1- alpha -> cost_matrix[i,j] = 100000 (set to invalid cost assignment value)
        """
        N = len(self.tracks)
        M = len(measurements)

        cost_matrix = np.zeros((N, M), dtype=float)
        
        for i, track in enumerate(self.tracks):
            for j, z in enumerate(measurements):
                d2 = self.mahalanobis_distance(track, z)
                if d2 <= self.gate_threshold:
                    cost_matrix[i,j] = d2
                else:
                    cost_matrix[i,j] = 1e6

        return cost_matrix

    def gnn_associate(self, measurements):
        """
        Build cost matrix and solve the linear solve assignment problem.
        
        The number of assignments in a nxm matrix where n neq m is min(n,m)
        Check to see is the assignment for a track is within gate threshold
        If so we assign tracks to measurements, mark unassigned tracks, and mark unassigned measurements.
        """
        cost_matrix = self.build_cost_matrix(measurements)

        row_ind, col_ind = linear_sum_assignment(cost_matrix) # Number of matches it will return is min(n,m) 

        assignments = []
        assigned_tracks = set()
        assigned_measurements = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.gate_threshold: # Even if a match is assigned throughout the cost_matrix it is dropped from assignment if it is invalid.
                assignments.append(
                    {
                        "track_id": self.tracks[i].track_id,
                        "measurment_number": j + 1,
                        "measurement": measurements[j]
                                    }
                                   )
                assigned_tracks.add(i)
                assigned_measurements.add(j)

        unassigned_tracks = [{"track_id": self.tracks[i].track_id}
                            for i in range(len(self.tracks))
                            if i not in assigned_tracks]
        unassigned_measurements = [{"measurment_number":j+1, "measurement": measurements[j]} for j in range(len(measurements)) if j not in assigned_measurements]

        return assignments, unassigned_tracks, unassigned_measurements, cost_matrix