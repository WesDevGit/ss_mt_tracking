import numpy as np

def pos_from_state(x):
    """Return [x, y] from a 4x1 state vector."""
    return np.asarray(x, dtype=float).reshape(-1)[:2]


def pred_key_to_truth_index(pred_key):
    """
    Map pred_log key to truth_state list index.

    'k'   -> 1   (truth at k)
    'k_1' -> 2   (truth at k+1)
    'k_2' -> 3   (truth at k+2)
    etc.
    """
    if pred_key == 'k':
        return 1
    return int(pred_key.split('_')[1]) + 1


def position_rmse_from_truth_and_predlog(truth_state, pred_log):
    """
    Compute overall position RMSE using:
      - truth_state[track_id] = [x_(k-1), x_k, x_(k+1), ...]
      - pred_log['k'][track_id] = [x_pred, P]
      - pred_log['k_1'][track_id] = [x_pred, P]
      - etc.

    Only uses matched samples that actually exist.
    """
    sq_errors = []
    matched_samples = []

    for pred_key, tracks_at_time in pred_log.items():
        truth_idx = pred_key_to_truth_index(pred_key)

        for track_id, pred_entry in tracks_at_time.items():
            if track_id not in truth_state:
                continue

            if truth_idx >= len(truth_state[track_id]):
                continue

            x_truth = truth_state[track_id][truth_idx]
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
        truth_idx = pred_key_to_truth_index(pred_key)

        for track_id in tracks_at_time:
            if track_id not in truth_state:
                continue
            if truth_idx >= len(truth_state[track_id]):
                continue
            matched += 1

    if total_possible == 0:
        return np.nan

    return matched / total_possible

def position_rmse_per_track_from_truth_and_predlog(truth_state, pred_log):
    per_track_sq_errors = {}

    for pred_key, tracks_at_time in pred_log.items():
        truth_idx = pred_key_to_truth_index(pred_key)

        for track_id, pred_entry in tracks_at_time.items():
            if track_id not in truth_state:
                continue

            if truth_idx >= len(truth_state[track_id]):
                continue

            x_truth = truth_state[track_id][truth_idx]
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