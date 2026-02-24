# Use an official lightweight Python/Linux image
FROM python:3.9-slim-buster

# Set system-level dependencies for OpenCV and ONNX
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
# Note: we'll use a minimal subset for inference
COPY requirements.txt .
RUN pip install --no-cache-dir ultralytics onnxruntime gradio numpy opencv-python-headless filterpy

# Copy source code
COPY . .

# Expose port for Gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app/main.py"]
