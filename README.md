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

## Papers:
@ARTICLE{2025arXiv250323108K,
       author = {{Kim}, Hyeongju and {Yang}, Jinhyeok and {Yu}, Yechan and {Ji}, Seunghun and {Morton}, Jacob and {Bous}, Frederik and {Byun}, Joon and {Lee}, Juheon},
        title = "{SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System}",
      journal = {arXiv e-prints},
     keywords = {Audio and Speech Processing, Machine Learning, Sound},
         year = 2025,
        month = mar,
          eid = {arXiv:2503.23108},
        pages = {arXiv:2503.23108},
          doi = {10.48550/arXiv.2503.23108},
archivePrefix = {arXiv},
       eprint = {2503.23108},
 primaryClass = {eess.AS},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250323108K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@misc{kim2025trainingflowmatchingmodels,
      title={Training Flow Matching Models with Reliable Labels via Self-Purification}, 
      author={Hyeongju Kim and Yechan Yu and June Young Yi and Juheon Lee},
      year={2025},
      eprint={2509.19091},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2509.19091}, 
}
@article{kim2025training,
  title={Training Flow Matching Models with Reliable Labels via Self-Purification},
  author={Kim, Hyeongju and Yu, Yechan and Yi, June Young and Lee, Juheon},
  journal={arXiv preprint arXiv:2509.19091},
  year={2025}
}
@misc{yi2025robustttstrainingselfpurifying,
      title={Robust TTS Training via Self-Purifying Flow Matching for the WildSpoof 2026 TTS Track}, 
      author={June Young Yi and Hyeongju Kim and Juheon Lee},
      year={2025},
      eprint={2512.17293},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.17293}, 
}
## License

MIT
