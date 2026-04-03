import matplotlib.pyplot as plt
import numpy as np

def plot_tracks(truth_positions, tracker):
    "Plot tracks based on truth position vectors and tracker object from track manager"
    plt.figure(figsize=(8, 6))

    # Plot truth trajectories
    for track_id, pos_list in truth_positions.items():
        x_vals = [pos[0, 0] for pos in pos_list]
        y_vals = [pos[1, 0] for pos in pos_list]
        plt.plot(x_vals, y_vals, marker='o', label=f"Truth Track {track_id}")

    # Collect predicted track positions across scans
    pred_positions = {}

    for scan in sorted(tracker.pred_log.keys()):
        for track_id, track_data in tracker.pred_log[scan].items():
            state = track_data[0]   # [state_vector, covariance_matrix]
            x = state[0, 0]
            y = state[1, 0]

            if track_id not in pred_positions:
                pred_positions[track_id] = {"x": [], "y": []}

            pred_positions[track_id]["x"].append(x)
            pred_positions[track_id]["y"].append(y)

    # Plot predicted trajectories
    for track_id, vals in pred_positions.items():
        plt.plot(
            vals["x"],
            vals["y"],
            marker='x',
            linestyle='--',
            label=f"Pred Track {track_id}"
        )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Truth and Predicted Trajectories")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()