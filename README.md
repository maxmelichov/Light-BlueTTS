# Light-BlueTTS

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

## Installation

```bash
uv sync                    # core deps
uv sync --extra gpu        # + CUDA support
```

## Model Weights

Download the TTS weights from [notmax123/LightBlue](https://huggingface.co/notmax123/LightBlue) and the Phonikud model from [thewh1teagle/phonikud-onnx](https://huggingface.co/thewh1teagle/phonikud-onnx).

```bash
uv run hf download notmax123/LightBlue --repo-type model --local-dir ./onnx_models
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
```

## Usage

```bash
uv run examples/basic.py
```

## Examples

See [examples](examples/)

## TensorRT

ONLY FOR NVIDIA GPUS!

```bash
uv run scripts/create_tensorrt.py --onnx_dir onnx_models --engine_dir trt_engines --precision fp16
uv run scripts/benchmark_trt.py --style_json voices/female1.json --steps 32
```

## License

MIT
