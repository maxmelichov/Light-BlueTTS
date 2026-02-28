#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hebrew_inference_helper import HebrewTTS, TTSConfig


def _try_load_z_ref(path: str) -> np.ndarray:
    if path.endswith(".pt"):
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(f"torch is required to load {path}: {e}")
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "z_ref_raw" in payload:
                z = payload["z_ref_raw"]
            elif "z_ref_norm" in payload:
                z = payload["z_ref_norm"]
            else:
                z = None
                for v in payload.values():
                    if torch.is_tensor(v):
                        z = v
                        break
                if z is None:
                    raise ValueError(f"Unsupported .pt payload keys: {list(payload.keys())}")
        else:
            z = payload
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        z = np.array(z, dtype=np.float32)
    else:
        z = np.load(path).astype(np.float32)
    if z.ndim == 2:
        z = z[None, :, :]
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="onnx_models")
    parser.add_argument("--config", default="tts.json", help="Path to tts.json config")
    parser.add_argument("--phonikud_path", default="phonikud-1.0.onnx")
    parser.add_argument("--z_ref", default=None)
    parser.add_argument("--style_json", default="voices/male1.json")
    parser.add_argument("--out", default="onnx_out.wav")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--chunk_len", type=int, default=150, help="Max chars per text chunk (default: 150)")
    parser.add_argument("--silence_sec", type=float, default=0.15, help="Silence inserted between chunks (default: 0.15)")
    args = parser.parse_args()

    if args.threads and int(args.threads) > 0:
        os.environ["ORT_INTRA"] = str(int(args.threads))
        os.environ["ORT_INTER"] = "1"

    default_text = (
        """jˈeʃ ʁeɡaʔˈim ʃebahˈem haʔolˈam ʔotsˈeʁ liʃnijˈa, lˈo kˈi kaʁˈa mˈaʃehu dʁamˈati,
        ʔˈela kˈi hevˈanu pitʔˈom mˈaʃehu χadˈaʃ. kˈaχ bedijˈuk niʁʔˈejt hatkufˈa ʃebˈa ʔˈanu χajˈim.
        ʁaʔjonˈot ʃenoldˈu kaχalˈom hofχˈim limetsiʔˈut ʃemeʔatsˈevet meχadˈaʃ ʔˈet hadˈeʁeχ ʃebˈa ʔˈanu
        medabʁˈim, jotsʁˈim umkablˈim haχlatˈot."""
    )
    text = args.text if args.text else default_text

    z_ref = _try_load_z_ref(args.z_ref) if args.z_ref else None
    style_json_path = args.style_json if (args.style_json and os.path.exists(args.style_json)) else None

    config = TTSConfig(
        onnx_dir=args.onnx_dir,
        config_path=args.config,
        phonikud_path=args.phonikud_path,
        use_gpu=not args.cpu,
        use_int8=args.int8,
        steps=args.steps,
        cfg_scale=args.cfg,
        speed=args.speed,
        seed=args.seed,
        text_chunk_len=args.chunk_len,
        silence_sec=args.silence_sec,
    )

    tts = HebrewTTS(config)

    t0 = time.time()
    wav = tts.infer(text, z_ref=z_ref, style_json_path=style_json_path)
    t1 = time.time()

    dur = len(wav) / float(tts.sample_rate) if len(wav) > 0 else 0.0
    if dur > 0:
        print(f"Generated {dur:.2f}s in {t1 - t0:.2f}s (RTF: {(t1 - t0) / dur:.3f})")
    else:
        print(f"Generated {dur:.2f}s in {t1 - t0:.2f}s")

    sf.write(args.out, wav, tts.sample_rate)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
