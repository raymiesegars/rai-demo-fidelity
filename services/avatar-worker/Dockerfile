FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY assets/ ./assets/

ENV PATIENT_LOOP_PATH=/workspace/assets/alan-loop.mp4
ENV TARGET_FPS=25
ENV AVATAR_MODE=mock
ENV LIVEKIT_ROOM=patient-demo

# LiveKit credentials passed at runtime
CMD ["python", "main.py"]
