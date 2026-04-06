import numpy as np
from scipy.optimize import linear_sum_assignment

def pos_from_state(x):
    """Return [x, y] from a 4x1 state vector."""
    return np.asarray(x, dtype=float).reshape(-1)[:2]


def nees_metrics(tracker, truth_data):
    truth_ids = {truth['id'] for truth in truth_data}
    all_nees = []

    for track_id, nees_values in tracker.update_log.items():
        if track_id in truth_ids and len(nees_values) > 0:
            print(f"Track_{track_id} NEES: {np.mean(nees_values)}")
            all_nees.extend(nees_values)

    print("------\n")

    if len(all_nees) > 0:
        print(f"Overall NEES: {np.mean(all_nees)}")
    else:
        print("Overall NEES: no values")

def nis_metrics(tracker, truth_data):
    truth_ids = {truth['id'] for truth in truth_data}
    all_nis = []

    for track_id, nis_values in tracker.consistency_log.items():
        if track_id in truth_ids and len(nis_values) > 0:
            print(f"Track_{track_id} NIS: {np.mean(nis_values)}")
            all_nis.extend(nis_values)

    print("------\n")

    if len(all_nis) > 0:
        print(f"Overall NIS: {np.mean(all_nis)}")
    else:
        print("Overall NIS: no values")
                
def position_rmse_from_truth_and_predlog(truth_state, pred_log):
    """
    Compute overall position RMSE using:
      - truth_state[track_id] = [x_(k-1), x_k, x_(k+1), ...]
      - pred_log['k'][track_id] = [x_pred, P]
      - pred_log['k_1'][track_id] = [x_pred, P]
      - etc.

    Only uses matched samples that actually exist.
    # This allows me to pull the proper index from each truth and prediction for alignment and look at the rmse
    # Not exactly sure if I should be taking all of truth with only what is captured.
    """
    sq_errors = []
    matched_samples = []

    for pred_key, tracks_at_time in pred_log.items():

        for track_id, pred_entry in tracks_at_time.items():
            if track_id not in truth_state:
                continue

            if pred_key >= len(truth_state[track_id]):
                continue

            x_truth = truth_state[track_id][pred_key]
            x_pred = pred_entry[0]   # pred_entry = [x_pred, P_pred]

            truth_xy = pos_from_state(x_truth)
            pred_xy  = pos_from_state(x_pred)

            err_vec = pred_xy - truth_xy
            sq_err = err_vec @ err_vec

            sq_errors.append(sq_err)
            matched_samples.append((pred_key, track_id, np.sqrt(sq_err)))

    if not sq_errors:
        return np.nan, [], 0

    rmse = np.sqrt(np.mean(sq_errors))
    return rmse, matched_samples, len(sq_errors)

def track_coverage_from_truth_and_predlog(truth_state, pred_log):
    total_possible = 0
    matched = 0

    for track_id, truth_list in truth_state.items():
        total_possible += max(len(truth_list) - 1, 0)

    for pred_key, tracks_at_time in pred_log.items():

        for track_id in tracks_at_time:
            if track_id not in truth_state:
                continue
            if pred_key >= len(truth_state[track_id]):
                continue
            matched += 1

    if total_possible == 0:
        return np.nan

    return matched / total_possible

def position_rmse_per_track_from_truth_and_predlog(truth_state, pred_log):
    per_track_sq_errors = {}

    for pred_key, tracks_at_time in pred_log.items():

        for track_id, pred_entry in tracks_at_time.items():
            if track_id not in truth_state:
                continue

            if pred_key >= len(truth_state[track_id]):
                continue

            x_truth = truth_state[track_id][pred_key]
            x_pred = pred_entry[0]

            truth_xy = pos_from_state(x_truth)
            pred_xy  = pos_from_state(x_pred)

            err_vec = pred_xy - truth_xy
            sq_err = err_vec @ err_vec

            if track_id not in per_track_sq_errors:
                per_track_sq_errors[track_id] = []

            per_track_sq_errors[track_id].append(sq_err)

    per_track_rmse = {}
    for track_id, sqerrs in per_track_sq_errors.items():
        if sqerrs:
            per_track_rmse[track_id] = np.sqrt(np.mean(sqerrs))
        else:
            per_track_rmse[track_id] = np.nan

    return per_track_rmse

def nees(x_hat, x, P):
    "Normalized estimation error squared (NEES)"
    epsilon = (x_hat - x).T @ np.linalg.solve(P, (x_hat - x))
    return epsilon


def _extract_state(entry):
    if isinstance(entry, dict):
        if "x" in entry:
            entry = entry["x"]
        elif "state" in entry:
            entry = entry["state"]
        else:
            raise ValueError(
                f"Could not find state inside dict entry keys={list(entry.keys())}"
            )
 
    if isinstance(entry, (list, tuple)):
        if len(entry) == 2 and _looks_like_covariance(entry[1]):
            entry = entry[0]
 
    return np.asarray(entry, dtype=float).reshape(-1, 1)
 
 
def _looks_like_covariance(obj):
    try:
        a = np.asarray(obj, dtype=float)
        return a.ndim == 2 and a.shape[0] == a.shape[1]
    except (ValueError, TypeError):
        return False
 
 
def _single_object_distance(x1, x2, components=(0, 1)):
    a = x1[list(components), 0]
    b = x2[list(components), 0]
    return float(np.linalg.norm(a - b))
 
def ospa_distance(X, Y, c=100.0, p=1, components=(0, 1)):
    m = len(X)
    n = len(Y)
 
    if m == 0 and n == 0:
        return 0.0, _empty_details(m, n, False)
 
    if m == 0 or n == 0:
        return float(c), _empty_details(m, n, False)
 
    swapped = False
    if m > n:
        X, Y = Y, X
        m, n = n, m
        swapped = True
 
    raw_D = np.zeros((m, n), dtype=float)
    capped_C_p = np.zeros((m, n), dtype=float)
 
    for i in range(m):
        for j in range(n):
            d = _single_object_distance(
                _extract_state(X[i]), _extract_state(Y[j]),
                components=components,
            )
            raw_D[i, j] = d
            capped_C_p[i, j] = min(c, d) ** p

    row_ind, col_ind = linear_sum_assignment(capped_C_p)
 
    localization_sum = float(np.sum(capped_C_p[row_ind, col_ind]))
    cardinality_sum = float((n - m) * (c ** p))
 
    ospa = ((localization_sum + cardinality_sum) / n) ** (1.0 / p)

    loc_component = float((localization_sum / n) ** (1.0 / p))
    card_component = float((cardinality_sum / n) ** (1.0 / p))
 
    details = {
        "m": m,
        "n": n,
        "swapped": swapped,
        "raw_distance_matrix": raw_D,
        "capped_cost_matrix_p": capped_C_p,
        "assigned_rows": row_ind,
        "assigned_cols": col_ind,
        "localization_sum_p": localization_sum,
        "cardinality_sum_p": cardinality_sum,
        "localization_component": loc_component,
        "cardinality_component": card_component,
    }
 
    return float(ospa), details
 
 
def _empty_details(m, n, swapped):
    """Return a minimal details dict for trivial (empty-set) cases."""
    return {
        "m": m,
        "n": n,
        "swapped": swapped,
        "raw_distance_matrix": np.empty((0, 0)),
        "capped_cost_matrix_p": np.empty((0, 0)),
        "assigned_rows": np.array([], dtype=int),
        "assigned_cols": np.array([], dtype=int),
        "localization_sum_p": 0.0,
        "cardinality_sum_p": 0.0,
        "localization_component": 0.0,
        "cardinality_component": 0.0,
    }
 
def truth_tracks_to_truth_by_scan(truth_states):
    track_ids = sorted(truth_states.keys())
    n_scans = max(len(truth_states[tid]) for tid in track_ids)
 
    truth_by_scan = {}
    for k in range(n_scans):
        scan_dict = {}
        for tid in track_ids:
            states = truth_states[tid]
            if k >= len(states):
                continue
            state = states[k]
            if state is None:
                continue
            scan_dict[tid] = _extract_state(state)
        truth_by_scan[k] = scan_dict
 
    return truth_by_scan
 
 
def ospa_from_truth_tracks_and_predlog(
    truth_states,
    pred_log,
    c=100.0,
    p=1,
    components=(0, 1, 2, 3),
):
    truth_by_scan = truth_tracks_to_truth_by_scan(truth_states)
    all_scan_keys = sorted(set(truth_by_scan.keys()) | set(pred_log.keys()))
 
    per_scan_ospa = {}
    per_scan_details = {}
 
    for scan_key in all_scan_keys:
        truth_scan = truth_by_scan.get(scan_key, {})
        pred_scan = pred_log.get(scan_key, {})
 
        X = [
            _extract_state(v)
            for _, v in sorted(truth_scan.items(), key=lambda kv: kv[0])
        ]
        Y = [
            _extract_state(v)
            for _, v in sorted(pred_scan.items(), key=lambda kv: kv[0])
        ]
        d_ospa, details = ospa_distance(X, Y, c=c, p=p, components=components)
        per_scan_ospa[scan_key] = d_ospa
        per_scan_details[scan_key] = details
 
    values = list(per_scan_ospa.values())
    avg_ospa = float(np.mean(values)) if values else 0.0
    median_ospa = float(np.median(values)) if values else 0.0
 
    return avg_ospa, median_ospa, per_scan_ospa, per_scan_details