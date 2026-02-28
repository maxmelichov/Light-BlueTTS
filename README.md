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

## Model Weights

Download the TTS weights from [notmax123/LightBlue](https://huggingface.co/notmax123/LightBlue) and the Phonikud model from [thewh1teagle/phonikud-onnx](https://huggingface.co/thewh1teagle/phonikud-onnx).

```bash
uv run hf download notmax123/LightBlue \
  --repo-type onnx_model \
  --local-dir ./onnx_model
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
```

Your project directory should look like this:

```
Light-BlueTTS/
├── phonikud-1.0.onnx              # from phonikud-onnx repo
└── onnx_models/
    ├── backbone.onnx               # flow-matching backbone
    ├── backbone_keys.onnx          # backbone with style_keys (for CFG)
    ├── text_encoder.onnx
    ├── reference_encoder.onnx
    ├── vocoder.onnx
    ├── length_pred.onnx            # duration predictor
    ├── length_pred_style.onnx      # duration predictor (style-conditioned)
    ├── stats.npz                   # normalization statistics
    └── uncond.npz                  # unconditional embeddings for CFG
```

## License

MIT
