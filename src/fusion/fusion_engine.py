import cv2
import numpy as np
from src.fusion.kalman_tracker import DroneTracker
from src.fusion.sensor_simulator import RF_Sensor_Simulator
from ultralytics import YOLO

class AeroFuseEngine:
    """
    The 'Wrapper' that fuses Visual AI, RF Signals, and Kalman Tracking.
    This is the core product logic for the DroneShield engineering task.
    """
    def __init__(self, model_path='AeroFuse/Phase-3/weights/best.pt'):
        self.model = YOLO(model_path)
        self.tracker = DroneTracker()
        self.rf_sim = RF_Sensor_Simulator()
        
    def process_mission(self, video_path, output_path='AeroFuse/Phase-4/demo_outputs/fused_output.mp4'):
        cap = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Prepare RF signals for the duration of the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        _, rf_signals = self.rf_sim.generate_signal(duration, [(2, 5)]) # Drone active between 2s and 5s
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.get()
            if not ret: break
            
            # 1. Visual Detection
            results = self.model(frame, verbose=False)[0]
            detection = None
            if len(results.boxes) > 0:
                # Get the center of the first detected drone
                box = results.boxes[0].xywh[0].cpu().numpy()
                detection = [box[0], box[1]]
            
            # 2. RF Signal Check
            rf_val = rf_signals[min(frame_idx, len(rf_signals)-1)]
            
            # 3. Sensor Fusion Logic
            # If AI is confident OR RF signal is high, we update/predict tracking
            if detection:
                self.tracker.update(detection)
                color = (0, 255, 0) # Green = Confident Visual
            elif rf_val > 0.6:
                color = (0, 165, 255) # Orange = Tracking via RF/Memory
            else:
                color = (0, 0, 255) # Red = Lost
                
            predicted_pos = self.tracker.predict()
            
            # 4. Visualization
            cv2.circle(frame, (int(predicted_pos[0]), int(predicted_pos[1])), 20, color, 2)
            cv2.putText(frame, f"RF Signal: {rf_val:.2f}", (50, 50), 1, 1, color, 2)
            
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"✅ Mission Processed. Fused output saved to {output_path}")

if __name__ == "__main__":
    # engine = AeroFuseEngine()
    # engine.process_mission('sample_drone.mp4')
    print("Fusion Engine ready. Requires a video file and trained weights to run.")
