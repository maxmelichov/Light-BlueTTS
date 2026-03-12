import os
import json
import re
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

from ._vocab import text_to_indices


class LightBlueTTS:
    def __init__(
        self,
        onnx_dir: str,
        config_path: str = "tts.json",
        style_json: Optional[str] = None,
        steps: int = 32,
        cfg_scale: float = 3.0,
        speed: float = 1.0,
        seed: int = 42,
        use_gpu: bool = False,
        chunk_len: int = 150,
        silence_sec: float = 0.15,
        fade_duration: float = 0.02,
    ):
        self.onnx_dir = onnx_dir
        self.style_json = style_json
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.chunk_len = chunk_len
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration

        self._load_config(config_path)
        self._init_sessions(use_gpu)
        self._load_stats()
        self._load_uncond()
        self._load_shuffle_keys()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str):
        self.normalizer_scale = 1.0
        self.latent_dim = 24
        self.chunk_compress_factor = 6
        self.hop_length = 512
        self.sample_rate = 44100

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.normalizer_scale = float(cfg.get("ttl", {}).get("normalizer", {}).get("scale", self.normalizer_scale))
            self.latent_dim = int(cfg.get("ttl", {}).get("latent_dim", self.latent_dim))
            self.chunk_compress_factor = int(cfg.get("ttl", {}).get("chunk_compress_factor", self.chunk_compress_factor))
            self.sample_rate = int(cfg.get("ae", {}).get("sample_rate", self.sample_rate))
            self.hop_length = int(cfg.get("ae", {}).get("encoder", {}).get("spec_processor", {}).get("hop_length", self.hop_length))

        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

    def _init_sessions(self, use_gpu: bool):
        available = ort.get_available_providers()
        if use_gpu:
            providers = [p for p in ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"] if p in available]
        else:
            providers = [p for p in ["OpenVINOExecutionProvider", "CPUExecutionProvider"] if p in available]

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_cores = max(1, (os.cpu_count() or 4) // 4)
        opts.intra_op_num_threads = int(os.environ.get("ORT_INTRA", cpu_cores))
        opts.inter_op_num_threads = int(os.environ.get("ORT_INTER", 1))

        self._opts = opts
        self._providers = providers

        self._text_enc = self._load_session("text_encoder.onnx")
        self._ref_enc = self._load_session("reference_encoder.onnx", required=False)

        vf_name = "backbone_keys.onnx" if os.path.exists(os.path.join(self.onnx_dir, "backbone_keys.onnx")) else "backbone.onnx"
        self._vf_model_name = vf_name.replace(".onnx", "")
        self._vf = self._load_session(vf_name)
        self._vocoder = self._load_session("vocoder.onnx")
        self._dp = self._load_session("length_pred.onnx", required=False)
        self._dp_style = self._load_session("length_pred_style.onnx", required=False)

        vf_inputs = {i.name for i in self._vf.get_inputs()}
        self._vf_inputs = vf_inputs
        self._vf_supports_style_keys = "style_keys" in vf_inputs
        self._vf_uses_text_emb = "text_emb" in vf_inputs and "text_context" not in vf_inputs

    def _load_session(self, name: str, required: bool = True) -> Optional[ort.InferenceSession]:
        base = os.path.join(self.onnx_dir, name)
        slim = base.replace(".onnx", ".slim.onnx")
        path = slim if os.path.exists(slim) else base
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Model not found: {base}")
            return None
        return ort.InferenceSession(path, sess_options=self._opts, providers=self._providers)

    def _load_stats(self):
        stats_path = os.path.join(self.onnx_dir, "stats.npz")
        self.mean = self.std = None
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)
            if self.mean.ndim == 1:
                self.mean = self.mean.reshape(1, -1, 1)
                self.std = self.std.reshape(1, -1, 1)
            if self.mean.ndim == 3:
                self.compressed_channels = int(self.mean.shape[1])
            if "normalizer_scale" in stats.files:
                self.normalizer_scale = float(stats["normalizer_scale"].item() if stats["normalizer_scale"].ndim == 0 else stats["normalizer_scale"][0])

    def _load_uncond(self):
        uncond_path = os.path.join(self.onnx_dir, "uncond.npz")
        self._u_text = self._u_ref = self._u_keys = self._cond_keys = None
        if os.path.exists(uncond_path):
            u = np.load(uncond_path)
            self._u_text = u["u_text"]
            self._u_ref = u["u_ref"]
            self._u_keys = u.get("u_keys") if "u_keys" in u.files else None
            self._cond_keys = u.get("cond_keys") if "cond_keys" in u.files else None

    def _load_shuffle_keys(self):
        self._model_keys: dict = {}
        keys_path = os.path.join(self.onnx_dir, "keys.npz")
        if not os.path.exists(keys_path):
            return
        data = np.load(keys_path)
        for k in data.files:
            parts = k.split("/", 1)
            if len(parts) == 2:
                model, inp = parts
                self._model_keys.setdefault(model, {})[inp] = data[k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, phonemes: str) -> Tuple[np.ndarray, int]:
        """Synthesize speech from pre-phonemized (IPA) text.

        Args:
            phonemes: IPA phoneme string (caller is responsible for phonemization).

        Returns:
            (samples, sample_rate) — float32 numpy array and int sample rate.
        """
        chunks = self._chunk(phonemes, self.chunk_len)
        silence = np.zeros(int(self.silence_sec * self.sample_rate), dtype=np.float32)
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(self._infer_chunk(chunk))
            if i < len(chunks) - 1:
                parts.append(silence)
        wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return wav, self.sample_rate

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, sess: ort.InferenceSession, feed: dict, model_name: str):
        keys = self._model_keys.get(model_name)
        if keys:
            feed = {**feed, **keys}
        return sess.run(None, feed)

    def _chunk(self, text: str, max_len: int) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
        pattern = r"(?<=[.!?])\s+"
        chunks = []
        for para in paragraphs:
            sentences = re.split(pattern, para)
            current = ""
            for s in sentences:
                if len(current) + len(s) + 1 <= max_len:
                    current += (" " if current else "") + s
                else:
                    if current:
                        chunks.append(current.strip())
                    current = s
            if current:
                chunks.append(current.strip())
        return chunks or [text]

    def _load_style_json(self, path: str):
        with open(path) as f:
            j = json.load(f)

        def _arr(key):
            if key not in j:
                return None
            a = np.array(j[key]["data"], dtype=np.float32)
            return a[None] if a.ndim == 2 else a

        style_ttl = _arr("style_ttl")
        style_keys = _arr("style_keys")
        style_dp = _arr("style_dp")
        z_ref = _arr("z_ref")
        return style_ttl, style_keys, style_dp, z_ref

    def _extract_style(self, z_ref_norm: np.ndarray):
        if self._ref_enc is None:
            raise ValueError("Reference encoder not loaded.")
        TARGET = 256
        B, C, T = z_ref_norm.shape
        if T < TARGET:
            pad = TARGET - T
            z = np.pad(z_ref_norm, ((0, 0), (0, 0), (0, pad)))
            mask = np.zeros((B, 1, TARGET), dtype=np.float32)
            mask[:, :, :T] = 1.0
        else:
            z = z_ref_norm[:, :, :TARGET]
            mask = np.ones((B, 1, TARGET), dtype=np.float32)

        ref_names = [i.name for i in self._ref_enc.get_inputs()]
        feed = {"z_ref": z}
        if "mask" in ref_names:
            feed["mask"] = mask
        elif "ref_mask" in ref_names:
            feed["ref_mask"] = mask
        elif len(ref_names) >= 2:
            feed[ref_names[1]] = mask

        ref_values, ref_keys = self._run(self._ref_enc, feed, "reference_encoder")[:2]
        return ref_values, ref_keys

    def _infer_chunk(self, phonemes: str) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("stats.npz not loaded.")

        style_ttl = style_keys = style_dp = z_ref = None
        if self.style_json:
            style_ttl, style_keys, style_dp, z_ref = self._load_style_json(self.style_json)

        if z_ref is None and style_ttl is None:
            raise ValueError("Provide style_json with z_ref or style_ttl content.")

        # Text encoding
        indices = text_to_indices(phonemes)
        text_ids = np.array([indices], dtype=np.int64)
        text_mask = np.ones((1, 1, len(indices)), dtype=np.float32)

        # Style
        z_ref_norm = None
        if z_ref is not None:
            z_ref_norm = ((z_ref - self.mean) / self.std) * float(self.normalizer_scale)
            T = z_ref_norm.shape[2]
            tail = max(2, int(T * 0.05))
            z_ref_norm = z_ref_norm[:, :, : max(1, T - tail)]
            if z_ref_norm.shape[2] > 150:
                z_ref_norm = z_ref_norm[:, :, :150]

        if style_ttl is not None:
            ref_values = style_ttl
        else:
            ref_values, style_keys = self._extract_style(z_ref_norm)

        if ref_values.ndim == 2:
            ref_values = ref_values[None]
        if style_keys is not None and style_keys.ndim == 2:
            style_keys = style_keys[None]

        ref_keys = style_keys if style_keys is not None else ref_values

        # Text encoder
        te_names = {i.name for i in self._text_enc.get_inputs()}
        te_feed = {"text_ids": text_ids}
        if "text_mask" in te_names:
            te_feed["text_mask"] = text_mask
        if "style_ttl" in te_names:
            te_feed["style_ttl"] = ref_values
        elif "ref_values" in te_names:
            te_feed["ref_values"] = ref_values
        else:
            raise ValueError("Unknown text encoder input names.")
        if "ref_keys" in te_names:
            te_feed["ref_keys"] = ref_keys
        elif "used_ref_keys" in te_names:
            te_feed["used_ref_keys"] = ref_keys

        text_emb = self._run(self._text_enc, te_feed, "text_encoder")[0]

        # Duration
        T_lat = self._predict_duration(text_ids, text_mask, z_ref_norm, style_dp)

        # Flow matching
        x = self._flow_matching(text_emb, ref_values, text_mask, T_lat)

        # Decode
        return self._decode(x)

    def _predict_duration(self, text_ids, text_mask, z_ref_norm, style_dp) -> int:
        T_lat = None

        if style_dp is not None and self._dp_style is not None:
            out = self._run(self._dp_style, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask}, "length_pred_style")
            val = float(np.squeeze(out[0]))
            if np.isfinite(val):
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None and z_ref_norm is not None and self._dp is not None:
            ref_len = int(z_ref_norm.shape[2])
            out = self._run(self._dp, {
                "text_ids": text_ids,
                "z_ref": z_ref_norm.astype(np.float32),
                "text_mask": text_mask,
                "ref_mask": np.ones((1, 1, ref_len), dtype=np.float32),
            }, "length_pred")
            val = float(np.squeeze(out[0]))
            if np.isfinite(val):
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None:
            T_lat = int(text_ids.shape[1] * 1.3)

        txt_len = int(np.sum(text_mask))
        T_cap = max(20, min(txt_len * 3 + 20, 600))
        T_lat = min(max(int(T_lat), 1), T_cap, 800)
        return max(10, T_lat)

    def _flow_matching(self, text_emb, ref_values, text_mask, T_lat) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        x = rng.randn(1, self.compressed_channels, T_lat).astype(np.float32)
        latent_mask = np.ones((1, 1, T_lat), dtype=np.float32)

        vf_inputs = self._vf_inputs
        cond_keys = None
        if self._vf_supports_style_keys and self._cond_keys is not None:
            cond_keys = self._cond_keys.astype(np.float32)
            if cond_keys.ndim == 2:
                cond_keys = cond_keys[None]

        u_text = self._u_text.astype(np.float32) if self._u_text is not None else None
        u_ref = self._u_ref.astype(np.float32) if self._u_ref is not None else None
        u_keys = self._u_keys.astype(np.float32) if self._u_keys is not None else None
        u_text_mask = np.ones((1, 1, 1), dtype=np.float32)

        for i in range(self.steps):
            t_val = np.array([float(i)], dtype=np.float32)
            total_t = np.array([float(self.steps)], dtype=np.float32)

            feed: dict = {}
            if "noisy_latent" in vf_inputs:
                feed["noisy_latent"] = x
            if "text_emb" in vf_inputs:
                feed["text_emb"] = text_emb
            elif "text_context" in vf_inputs:
                feed["text_context"] = text_emb
            if "style_ttl" in vf_inputs:
                feed["style_ttl"] = ref_values
            elif "ref_values" in vf_inputs:
                feed["ref_values"] = ref_values
            if "latent_mask" in vf_inputs:
                feed["latent_mask"] = latent_mask
            if "text_mask" in vf_inputs:
                feed["text_mask"] = text_mask
            if "current_step" in vf_inputs:
                feed["current_step"] = t_val
            if "total_step" in vf_inputs:
                feed["total_step"] = total_t
            if "style_keys" in vf_inputs and cond_keys is not None:
                feed["style_keys"] = cond_keys
            if "style_mask" in vf_inputs:
                feed["style_mask"] = np.ones((1, 1, ref_values.shape[1]), dtype=np.float32)

            den_cond = self._run(self._vf, feed, self._vf_model_name)[0]

            if self.cfg_scale != 1.0 and u_text is not None:
                feed_u = dict(feed)
                if "text_emb" in vf_inputs:
                    feed_u["text_emb"] = u_text
                elif "text_context" in vf_inputs:
                    feed_u["text_context"] = u_text
                if "style_ttl" in vf_inputs:
                    feed_u["style_ttl"] = u_ref
                elif "ref_values" in vf_inputs:
                    feed_u["ref_values"] = u_ref
                if "text_mask" in vf_inputs:
                    feed_u["text_mask"] = u_text_mask
                if "style_keys" in vf_inputs:
                    feed_u["style_keys"] = u_keys
                if "style_mask" in vf_inputs:
                    feed_u["style_mask"] = np.ones((1, 1, u_ref.shape[1]), dtype=np.float32)

                den_uncond = self._run(self._vf, feed_u, self._vf_model_name)[0]
                x = den_uncond + self.cfg_scale * (den_cond - den_uncond)
            else:
                x = den_cond

        return x

    def _apply_fade(self, wav: np.ndarray) -> np.ndarray:
        fade_samples = int(self.fade_duration * self.sample_rate)
        if fade_samples == 0 or len(wav) < 2 * fade_samples:
            return wav
        wav = wav.copy()
        wav[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        return wav

    def _decode(self, z_pred: np.ndarray) -> np.ndarray:
        if float(self.normalizer_scale) not in (0.0, 1.0):
            z_unnorm = (z_pred / float(self.normalizer_scale)) * self.std + self.mean
        else:
            z_unnorm = z_pred * self.std + self.mean

        B, C, T = z_unnorm.shape
        z_dec = (
            z_unnorm.reshape(B, self.latent_dim, self.chunk_compress_factor, T)
            .transpose(0, 1, 3, 2)
            .reshape(B, self.latent_dim, T * self.chunk_compress_factor)
        )

        wav = self._run(self._vocoder, {"latent": z_dec}, "vocoder")[0]

        frame_len = int(self.hop_length * self.chunk_compress_factor)
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]

        wav = wav.squeeze()
        return self._apply_fade(wav)
