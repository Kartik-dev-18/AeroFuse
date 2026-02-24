import torch
from ultralytics import YOLO

def export_optimized(model_path='AeroFuse/Phase-3/weights/best.pt'):
    """
    Advanced Export: Includes ONNX and potential for Int8 Quantization.
    Quantization makes the model 4x smaller and significantly faster on Edge CPUs.
    """
    try:
        model = YOLO(model_path)
        print(f"🚀 Starting Edge-Optimization for {model_path}...")
        
        # Standard ONNX
        onnx_path = model.export(format='onnx', opset=12, simplify=True)
        print(f"✅ Standard ONNX Exported: {onnx_path}")
        
        # INT8 Quantization (The 'Pro' Move)
        # Note: This requires a calibration dataset in a real scenario
        print("💡 Pro-Tip: For DroneShield, you would mention 'Post-Training Quantization' here.")
        # openvino_path = model.export(format='openvino', int8=True) # Great for Intel Edge
        
    except Exception as e:
        print(f"Export Error: {e}")

if __name__ == "__main__":
    export_optimized()
