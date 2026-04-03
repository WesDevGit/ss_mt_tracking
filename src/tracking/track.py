
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
        