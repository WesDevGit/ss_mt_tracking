import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_covariance_ellipse(
    P_xy,
    center,
    confidence=0.95,
    ax=None,
    alpha=0.15,
    linewidth=1.0,
):
    """
    Plot a 2D covariance ellipse for the 2x2 position covariance block.
    """
    if ax is None:
        ax = plt.gca()

    P_xy = np.asarray(P_xy, dtype=float)

    # Symmetric eigendecomposition for covariance
    eigvals, eigvecs = np.linalg.eigh(P_xy)

    # Protect against tiny negative eigenvalues from numerical roundoff
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    # Sort largest first
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Chi-square scale for chosen confidence in 2D
    scale = chi2.ppf(confidence, df=2)
    width = 2.0 * np.sqrt(eigvals[0] * scale)
    height = 2.0 * np.sqrt(eigvals[1] * scale)

    # Angle of major axis
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ell = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        fill=True,
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.add_patch(ell)
    return ell


def rb_to_xy(z, sensor_position):
    """
    Convert range-bearing measurement z=[[r],[theta]] to Cartesian position.
    """
    r = z[0, 0]
    b = z[1, 0]
    sx, sy = sensor_position.flatten()

    px = sx + r * np.cos(b)
    py = sy + r * np.sin(b)

    return np.array([[px], [py]], dtype=float)


def plot_tracks(
    truth_positions,
    tracker,
    scans=None,
    sensor_position=None,
    confidence=0.95,
    ellipse_every=1,
    ellipse_alpha=0.12,
    show_scan_labels=False,
    show_measurements=False,
):
    """
    Plot truth tracks, predicted tracks, and covariance ellipses.

    Parameters
    ----------
    truth_positions : dict
        {track_id: [np.array([[px], [py]]), ...]}

    tracker : TrackManager-like object
        Must contain tracker.pred_log where:
        tracker.pred_log[scan][track_id] = [state_vector, covariance_matrix]

    scans : dict, optional
        If provided and show_measurements=True, expects:
        scans[scan_key] = list of measurement vectors [[r], [theta]]

    sensor_position : np.ndarray, optional
        Required if show_measurements=True

    ellipse_every : int
        Plot ellipse every N scans per track to reduce clutter.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Truth trajectories
    for track_id, pos_list in truth_positions.items():
        x_vals = [pos[0, 0] for pos in pos_list]
        y_vals = [pos[1, 0] for pos in pos_list]
        ax.plot(x_vals, y_vals, marker="o", label=f"Truth Track {track_id}")

    # Predicted trajectories and ellipses
    pred_positions = {}
    pred_scan_keys = list(tracker.pred_log.keys())

    for scan_idx, scan_key in enumerate(pred_scan_keys):
        for track_id, track_data in tracker.pred_log[scan_key].items():
            x_state = track_data[0]
            P = track_data[1]

            px = x_state[0, 0]
            py = x_state[1, 0]

            if track_id not in pred_positions:
                pred_positions[track_id] = {"x": [], "y": [], "scan": []}

            pred_positions[track_id]["x"].append(px)
            pred_positions[track_id]["y"].append(py)
            pred_positions[track_id]["scan"].append(scan_key)

            # Plot covariance ellipse in position space
            if ellipse_every is not None and scan_idx % ellipse_every == 0:
                P_xy = P[:2, :2]
                plot_covariance_ellipse(
                    P_xy,
                    center=(px, py),
                    confidence=confidence,
                    ax=ax,
                    alpha=ellipse_alpha,
                    linewidth=1.0,
                )

            if show_scan_labels:
                ax.text(px, py, str(scan_key), fontsize=8)

    # Predicted trajectories
    for track_id, vals in pred_positions.items():
        if len(vals['x']) < 5:
            continue
        ax.plot(
            vals["x"],
            vals["y"],
            marker="x",
            linestyle="--",
            label=f"Pred Track {track_id}",
        )

    # Optional measurement plot
    if show_measurements:
        if scans is None or sensor_position is None:
            raise ValueError(
                "scans and sensor_position are required when show_measurements=True"
            )

        meas_x = []
        meas_y = []

        for scan_key in scans.keys():
            for z in scans[scan_key]:
                xy = rb_to_xy(z, sensor_position)
                meas_x.append(xy[0, 0])
                meas_y.append(xy[1, 0])

        ax.scatter(meas_x, meas_y, marker=".", alpha=0.5, label="Measurements")

    # Optional sensor marker
    if sensor_position is not None:
        sx, sy = sensor_position.flatten()
        ax.scatter([sx], [sy], marker="s", s=80, label="Sensor")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Truth, Predicted Tracks, and Covariance Ellipses")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.show()