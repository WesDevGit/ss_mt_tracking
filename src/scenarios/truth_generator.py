import numpy as np
from src.models import measurement_models
from src.models import motion_models
from configs.multi_track_ss_baseline import measurement_noise
def generate_truth(n_steps, truth_data, P, id_miss_index, R, dt):
    true_state = []
    track_truths = [] # truth starts from k-1
    measure_data = [] # measurements start from k
    for state in truth_data:
        x_k = state['x'].copy()
        truth_states = {"id": state['id'], "x_states": [x_k.copy()], "P": P}
        track_measurements = {"id": state['id']}
        for _ in range(n_steps):
            x_k = motion_models.F @ x_k
            truth_states['x_states'].append(x_k.copy())
            if not 'measurements' in track_measurements.keys():
                track_measurements['measurements'] = [measurement_models.H @ x_k.copy() + measurement_noise(R)]
            else:
                track_measurements['measurements'].append(measurement_models.H @ x_k.copy() + measurement_noise(R))
        track_truths.append(truth_states)
        measure_data.append(track_measurements)

    truth_states = {}
    for tid in track_truths: # truth starts from k-1 to k99 so you have 101 
        truth_states[tid['id']] =  [x for x in tid['x_states']]
    truth_positions = {}
    for tid in track_truths:
        truth_positions[tid['id']] =  [x[:2,:] for x in tid['x_states']]
    truth_velocities = {}
    for tid in track_truths:
        truth_velocities[tid['id']] =  [x[2:,:] for x in tid['x_states']]
    truth_times = [i * dt for i in range(n_steps + 1)] # k-1 to k_99 = 101 
    scans = build_scans(measure_data, id_miss_index)
    truth_exists = truth_misses(track_truths, id_miss_index, len(truth_times))
    return truth_states, truth_positions, truth_velocities, truth_times, truth_exists, scans

### Truth data is a list of dictionaries with {"id":, "x", "P"}
 
def build_scans(measure_data, miss_indices, clutter_rate=0.0, clutter_bounds=None):
    all_measurements = [md['measurements'] for md in measure_data]
    id_to_idx = {md['id']: i for i, md in enumerate(measure_data)}
    pos_miss = {id_to_idx[tid]: indices for tid, indices in miss_indices.items()
                if tid in id_to_idx}

    n_scans = len(all_measurements[0])
    scans = {}

    for i in range(n_scans):
        scan_index = i + 1
        scan_measurements = []
        for track_idx, track_meas in enumerate(all_measurements):
            if track_idx in pos_miss and scan_index in pos_miss[track_idx]:
                continue
            scan_measurements.append(track_meas[i])

        if clutter_rate > 0 and clutter_bounds is not None:
            n_clutter = np.random.poisson(clutter_rate)
            for _ in range(n_clutter):
                false_alarm = np.array([
                    [np.random.uniform(clutter_bounds['x_min'], clutter_bounds['x_max'])],
                    [np.random.uniform(clutter_bounds['y_min'], clutter_bounds['y_max'])]
                ])
                scan_measurements.append(false_alarm)

        scans[scan_index] = scan_measurements

    return scans

def truth_misses(track_truths, id_miss_index, n_times):
    truth_exists = {}
    for tid in track_truths:
        truth_exists[tid['id']] = [1] * n_times
        if tid['id'] in id_miss_index:
            for idx in id_miss_index[tid['id']]:
                truth_exists[tid['id']][idx] = 0
    return truth_exists
