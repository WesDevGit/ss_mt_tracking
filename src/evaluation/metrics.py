import numpy as np
from scipy.optimize import linear_sum_assignment


def pos_from_state(x):
    return np.asarray(x, dtype=float).reshape(-1)[:2]


def nees(x_hat, x, P):
    e = x_hat - x
    return float(e.T @ np.linalg.solve(P, e))


def _looks_like_covariance(obj):
    try:
        a = np.asarray(obj, dtype=float)
        return a.ndim == 2 and a.shape[0] == a.shape[1]
    except (ValueError, TypeError):
        return False


def _extract_state(entry):
    if isinstance(entry, dict):
        if "x" in entry:
            entry = entry["x"]
        elif "state" in entry:
            entry = entry["state"]
        else:
            raise ValueError(f"Could not find state in dict keys={list(entry.keys())}")

    if isinstance(entry, (list, tuple)) and len(entry) == 2 and _looks_like_covariance(entry[1]):
        entry = entry[0]

    return np.asarray(entry, dtype=float).reshape(-1, 1)


def _summarize_track_metric(log_dict, truth_data, name):
    truth_ids = {truth["id"] for truth in truth_data}
    all_values = []

    for track_id, values in log_dict.items():
        if track_id in truth_ids and values:
            mean_val = float(np.mean(values))
            print(f"Track_{track_id} {name}: {mean_val}")
            all_values.extend(values)

    print("------\n")
    if all_values:
        print(f"Overall {name}: {float(np.mean(all_values))}")
    else:
        print(f"Overall {name}: no values")


def nees_metrics(tracker, truth_data):
    _summarize_track_metric(tracker.update_log, truth_data, "NEES")


def nis_metrics(tracker, truth_data):
    _summarize_track_metric(tracker.consistency_log, truth_data, "NIS")


def _matched_truth_pred_pairs(truth_state, pred_log):
    for scan_key, tracks_at_time in pred_log.items():
        for track_id, pred_entry in tracks_at_time.items():
            if track_id not in truth_state:
                continue
            if scan_key >= len(truth_state[track_id]):
                continue

            x_truth = truth_state[track_id][scan_key]
            x_pred = pred_entry[0]

            yield scan_key, track_id, _extract_state(x_truth), _extract_state(x_pred)


def position_rmse_from_truth_and_predlog(truth_state, pred_log):
    sq_errors = []
    matched_samples = []

    for scan_key, track_id, x_truth, x_pred in _matched_truth_pred_pairs(truth_state, pred_log):
        err = pos_from_state(x_pred) - pos_from_state(x_truth)
        sq_err = float(err @ err)
        sq_errors.append(sq_err)
        matched_samples.append((scan_key, track_id, np.sqrt(sq_err)))

    if not sq_errors:
        return np.nan, [], 0

    return float(np.sqrt(np.mean(sq_errors))), matched_samples, len(sq_errors)


def position_rmse_per_track_from_truth_and_predlog(truth_state, pred_log):
    per_track_sq_errors = {}

    for _, track_id, x_truth, x_pred in _matched_truth_pred_pairs(truth_state, pred_log):
        err = pos_from_state(x_pred) - pos_from_state(x_truth)
        sq_err = float(err @ err)
        per_track_sq_errors.setdefault(track_id, []).append(sq_err)

    return {
        track_id: float(np.sqrt(np.mean(sqerrs))) if sqerrs else np.nan
        for track_id, sqerrs in per_track_sq_errors.items()
    }


def track_coverage_from_truth_and_predlog(truth_state, pred_log):
    total_possible = sum(max(len(states) - 1, 0) for states in truth_state.values())
    matched = sum(1 for _ in _matched_truth_pred_pairs(truth_state, pred_log))
    return np.nan if total_possible == 0 else matched / total_possible


def _single_object_distance(x1, x2, components=(0, 1)):
    a = x1[list(components), 0]
    b = x2[list(components), 0]
    return float(np.linalg.norm(a - b))


def _empty_ospa_details(m, n, swapped):
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


def ospa_distance(X, Y, c=100.0, p=1, components=(0, 1)):
    X = [_extract_state(x) for x in X]
    Y = [_extract_state(y) for y in Y]
    m, n = len(X), len(Y)

    if m == 0 and n == 0:
        return 0.0, _empty_ospa_details(m, n, False)

    if m == 0 or n == 0:
        return float(c), _empty_ospa_details(m, n, False)

    swapped = False
    if m > n:
        X, Y = Y, X
        m, n = n, m
        swapped = True

    raw_D = np.zeros((m, n), dtype=float)
    cost_p = np.zeros((m, n), dtype=float)

    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            d = _single_object_distance(xi, yj, components=components)
            raw_D[i, j] = d
            cost_p[i, j] = min(c, d) ** p

    row_ind, col_ind = linear_sum_assignment(cost_p)

    localization_sum = float(np.sum(cost_p[row_ind, col_ind]))
    cardinality_sum = float((n - m) * (c ** p))

    ospa = ((localization_sum + cardinality_sum) / n) ** (1.0 / p)
    loc_component = (localization_sum / n) ** (1.0 / p)
    card_component = (cardinality_sum / n) ** (1.0 / p)

    details = {
        "m": m,
        "n": n,
        "swapped": swapped,
        "raw_distance_matrix": raw_D,
        "capped_cost_matrix_p": cost_p,
        "assigned_rows": row_ind,
        "assigned_cols": col_ind,
        "localization_sum_p": localization_sum,
        "cardinality_sum_p": cardinality_sum,
        "localization_component": float(loc_component),
        "cardinality_component": float(card_component),
    }
    return float(ospa), details


def truth_tracks_to_truth_by_scan(truth_states):
    track_ids = sorted(truth_states.keys())
    n_scans = max(len(truth_states[tid]) for tid in track_ids)

    return {
        k: {
            tid: _extract_state(truth_states[tid][k])
            for tid in track_ids
            if k < len(truth_states[tid]) and truth_states[tid][k] is not None
        }
        for k in range(n_scans)
    }


def ospa_from_truth_tracks_and_predlog(truth_states, pred_log, c=100.0, p=1, components=(0, 1, 2, 3)):
    truth_by_scan = truth_tracks_to_truth_by_scan(truth_states)
    all_scan_keys = sorted(set(truth_by_scan) | set(pred_log))

    per_scan_ospa = {}
    per_scan_details = {}

    for scan_key in all_scan_keys:
        truth_scan = truth_by_scan.get(scan_key, {})
        pred_scan = pred_log.get(scan_key, {})

        X = [v for _, v in sorted(truth_scan.items())]
        Y = [_extract_state(v) for _, v in sorted(pred_scan.items())]

        d_ospa, details = ospa_distance(X, Y, c=c, p=p, components=components)
        per_scan_ospa[scan_key] = d_ospa
        per_scan_details[scan_key] = details

    values = list(per_scan_ospa.values())
    avg_ospa = float(np.mean(values)) if values else 0.0
    median_ospa = float(np.median(values)) if values else 0.0

    return avg_ospa, median_ospa, per_scan_ospa, per_scan_details