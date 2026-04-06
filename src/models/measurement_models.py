import numpy as np
# constant velocity
H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
], dtype=float)


# I will receive a new measurement which will be range and bearing.

def h_range_az(predicted_state, sensor_position):
    px, py, vx, py, omega = predicted_state.flatten()

    sx, sy = sensor_position.flatten()
    
    dx, dy = (px - sx), (py - sy)
    r = np.sqrt(dx**2 + dy**2)
    h_x = np.array([[r],
                    [np.arctan2(dy, dx)]])
    
    H = np.array([[dx/r,       dy/r,      0, 0, 0],
                  [-dy/(r**2), dx/(r**2), 0, 0, 0]])
    return h_x, H