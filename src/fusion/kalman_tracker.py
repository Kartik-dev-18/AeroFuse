import numpy as np
from filterpy.kalman import KalmanFilter

class DroneTracker:
    """
    A Kalman Filter based tracker for drone position (x, y).
    This handles 'State Estimation' - a core part of sensor fusion.
    """
    def __init__(self, dt=0.1):
        # 4 states: [x, y, dx, dy] (position and velocity)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        # Measurement function (we only measure x and y)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        # Covariance matrices (tuning these is the 'engineering' part)
        self.kf.P *= 10.0
        self.kf.R = np.eye(2) * 5.0 # Measurement noise
        self.kf.Q = np.eye(4) * 0.1 # Process noise

    def update(self, measurement):
        """Update the filter with a new visual detection [x, y]"""
        if measurement is not None:
            self.kf.update(measurement)

    def predict(self):
        """Predict the next state. Useful during frame-drops or occlusions."""
        self.kf.predict()
        return self.kf.x[:2].flatten()

    def get_state(self):
        return self.kf.x

if __name__ == "__main__":
    tracker = DroneTracker()
    # Simulate a few steps
    tracker.update([100, 100])
    print(f"Prediction: {tracker.predict()}")
