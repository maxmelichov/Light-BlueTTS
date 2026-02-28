FROM python:3.10-slim

WORKDIR /app

# System deps required by soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install python deps
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Copy models + assets + code
COPY onnx_models /app/onnx_models
COPY phonikud-1.0.onnx /app/phonikud-1.0.onnx
COPY voices /app/voices
COPY tts.json /app/tts.json
COPY *.npy /app/

COPY utils.py /app/utils.py
COPY text_vocab.py /app/text_vocab.py
COPY hebrew_inference_helper.py /app/hebrew_inference_helper.py
COPY run_onnx_inference.py /app/run_onnx_inference.py

# Runtime environment
ENV ONNX_DIR=/app/onnx_models \
    PHONIKUD_PATH=/app/phonikud-1.0.onnx \
    DEFAULT_STYLE_JSON=/app/voices/male1.json \
    SAMPLE_RATE=44100 \
    PYTHONUNBUFFERED=1 \
    ORT_DISABLE_CPU_AFFINITY=1 \
    ORT_INTRA=4 \
    ORT_INTER=1

CMD ["uv", "run", "python", "-u", "run_onnx_inference.py"]
