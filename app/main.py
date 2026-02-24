import gradio as gr
import os
from src.fusion.fusion_engine import AeroFuseEngine

# Initialize the engine
# Note: In HF Spaces, we'll place the weight in the root or weights/ folder
weights_path = 'best.onnx' 

def run_fused_inference(video_input, rf_sensitivity):
    """
    Main entry point for the web demo.
    """
    if video_input is None:
        return "Please upload a video file."
        
    engine = AeroFuseEngine(model_path=weights_path)
    output_video_path = 'output_demo.mp4'
    
    # Process the video with the real fusion logic
    engine.process_mission(video_input, output_path=output_video_path)
    
    return output_video_path

# UI Styling
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(theme=theme, title="AeroFuse Demo") as demo:
    gr.Markdown("""
    # 🛸 AeroFuse: Multi-Sensor Drone Tracking
    ### Mission-Critical Edge Fusion for DroneShield AI/ML Portfolio
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="UAV Video Feed")
            rf_slider = gr.Slider(0, 1, value=0.7, label="RF Signal Noise Floor")
            submit_btn = gr.Button("🚀 Start Fusion Tracking", variant="primary")
            
        with gr.Column(scale=1):
            video_out = gr.Video(label="Fused System Output (AI + RF + Kalman)")
            
    gr.Markdown("""
    **System Architecture Note:** This demo utilizes an **ONNX-optimized YOLOv8n** model for detection 
    fused with a **Discrete Kalman Filter** for state estimation. During visual signal degradation 
    (low AI confidence), the system maintains track integrity via the RF signal threshold.
    """)

if __name__ == "__main__":
    demo.launch()

if __name__ == "__main__":
    demo.launch()
