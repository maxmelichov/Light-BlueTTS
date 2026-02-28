# Light-BlueTTS

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

## Installation

```bash
uv sync                    # core deps
uv sync --extra gpu        # + CUDA support
uv sync --extra tensorrt   # + TensorRT
```

## Usage

```bash
uv run python run_onnx_inference.py \
    --text "שלום עולם" --style_json voices/female1.json --out output.wav
```

```python
from hebrew_inference_helper import HebrewTTS, TTSConfig

tts = HebrewTTS(TTSConfig(onnx_dir="onnx_models", config_path="tts.json"))
wav = tts.infer("שלום עולם", style_json_path="voices/female1.json")
```

## TensorRT
ONLY FOR NVIDIA GPUS!

```bash
uv run python create_tensorrt.py --onnx_dir onnx_models --engine_dir trt_engines --precision fp16
uv run python benchmark_trt.py --style_json voices/female1.json --steps 32
uv run python test_gpu_onnx.py --compare --runs 3
```

## Required Models

Place in `onnx_models/`: `text_encoder.onnx`, `vector_estimator.onnx`, `vocoder.onnx`, `duration_predictor.onnx`, `stats.npz`, `uncond.npz`. Also needed: [`phonikud-1.0.onnx`](https://huggingface.co/thewh1teagle/phonikud-onnx/blob/main/phonikud-1.0.onnx).

## License

MIT
