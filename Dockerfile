# Before building:
#
#   1. Download model weights:
#      https://huggingface.co/notmax123/LightBlue       -> onnx_models/, voices/, tts.json
#      https://huggingface.co/thewh1teagle/phonikud-onnx -> phonikud-1.0.onnx
#
#   2. Build and run:
#      docker build -t lightblue-tts .
#      docker run --rm lightblue-tts --text "שלום עולם"

FROM python:3.10-slim

WORKDIR /app

# System deps required by soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install python deps (package + app-level phonemization deps)
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project
RUN uv pip install phonikud phonikud-onnx

# Copy package + app + models + assets
COPY src/ /app/src/
COPY examples/ /app/examples/
COPY onnx_models /app/onnx_models
COPY phonikud-1.0.onnx /app/phonikud-1.0.onnx
COPY voices /app/voices
COPY tts.json /app/tts.json

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    ORT_DISABLE_CPU_AFFINITY=1 \
    ORT_INTRA=4 \
    ORT_INTER=1

CMD ["uv", "run", "python", "-u", "examples/app.py"]
