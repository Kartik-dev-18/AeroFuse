import numpy as np

class RF_Sensor_Simulator:
    """
    Simulates a Radio Frequency (RF) sensor that detects drone signals.
    In a real scenario like DroneShield, this would be a software-defined radio (SDR) 
    picking up 2.4GHz or 5.8GHz hops.
    """
    def __init__(self, sample_rate=10):
        self.sample_rate = sample_rate # Samples per second

    def generate_signal(self, duration_sec, drone_present_intervals):
        """
        Generates a signal strength array.
        drone_present_intervals: List of tuples (start_sec, end_sec)
        """
        num_samples = int(duration_sec * self.sample_rate)
        time = np.linspace(0, duration_sec, num_samples)
        
        # Base noise floor
        signal = np.random.normal(0.1, 0.05, num_samples)
        
        for start, end in drone_present_intervals:
            mask = (time >= start) & (time <= end)
            # Add a spike with some variance when drone is present
            signal[mask] += np.random.normal(0.8, 0.1, np.sum(mask))
            
        return time, np.clip(signal, 0, 1.2)

if __name__ == "__main__":
    # Quick test
    sim = RF_Sensor_Simulator()
    t, s = sim.generate_signal(10, [(2, 5)])
    print(f"Generated {len(s)} RF samples.")
    # In practice, we'll use this to sync with video frames
