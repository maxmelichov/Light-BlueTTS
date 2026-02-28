import os
import sys
import numpy as np
import onnxruntime as ort
import argparse
import time
import json
import re
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict

# Phonikud imports
try:
    from phonikud_onnx import Phonikud
    from phonikud import phonemize
except ImportError:
    Phonikud = None
    phonemize = None

# Text vocab
from text_vocab import text_to_indices

@dataclass
class TTSConfig:
    """Configuration for HebrewTTS"""
    # Model Paths
    onnx_dir: str = "onnx_models"
    config_path: str = "tts.json"
    phonikud_path: str = "phonikud-1.0.onnx"
    use_gpu: bool = False
    use_int8: bool = False

    # Inference Parameters
    steps: int = 32
    cfg_scale: float = 3.0
    speed: float = 1.0
    seed: int = 42

    # Audio Settings
    sample_rate: int = 44100
    silence_sec: float = 0.15
    fade_duration: float = 0.02

    # Text / Reference handling
    text_chunk_len: int = 150
    ref_target_frames: int = 150

class HebrewTTS:
    def __init__(self, config: TTSConfig):
        self.config = config
        
        self._load_cfg()

        # 1. Initialize Phonikud (optional)
        if Phonikud is not None and phonemize is not None and os.path.exists(config.phonikud_path):
            print(f"Loading Phonikud from {config.phonikud_path}...")
            self.phonikud = Phonikud(config.phonikud_path)
        else:
            self.phonikud = None
        
        # 2. Initialize ONNX Sessions
        self._init_onnx_sessions()
        
        # 3. Load Stats
        self._load_stats()
        
        # 4. Load Unconditional Tokens (for CFG)
        self._load_uncond()

        # 5. Load Shuffle Keys (for optimizer-proof obfuscated models)
        self._load_shuffle_keys()

    def _load_cfg(self):
        self.normalizer_scale = 1.0
        self.latent_dim = 24
        self.chunk_compress_factor = 6
        self.hop_length = 512
        self.sample_rate = int(self.config.sample_rate)

        cfg_path = self.config.config_path
        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                self.normalizer_scale = float(cfg.get("ttl", {}).get("normalizer", {}).get("scale", self.normalizer_scale))
                self.latent_dim = int(cfg.get("ttl", {}).get("latent_dim", self.latent_dim))
                self.chunk_compress_factor = int(cfg.get("ttl", {}).get("chunk_compress_factor", self.chunk_compress_factor))
                self.sample_rate = int(cfg.get("ae", {}).get("sample_rate", self.sample_rate))
                self.hop_length = int(cfg.get("ae", {}).get("encoder", {}).get("spec_processor", {}).get("hop_length", self.hop_length))
            except Exception:
                pass

        self.compressed_channels = int(self.latent_dim * self.chunk_compress_factor)

    def _init_onnx_sessions(self):
        available_providers = ort.get_available_providers()
        print(f"[INFO] Available ORT Providers: {available_providers}")
        
        if self.config.use_gpu:
            providers = [p for p in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]
        else:
            providers = [p for p in ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]
            
        print(f"Loading ONNX models from {self.config.onnx_dir} with providers={providers}...")
        
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Threading strategy
        intra_def = os.environ.get("ORT_INTRA")
        inter_def = os.environ.get("ORT_INTER", "1")
        cpu_cores = os.cpu_count() // 4
        
        opts.intra_op_num_threads = int(intra_def) if intra_def else cpu_cores
        opts.inter_op_num_threads = int(inter_def)
        print(f"[INFO] ORT Threads: intra={opts.intra_op_num_threads}, inter={opts.inter_op_num_threads}")
        
        self.sess_opts = opts
        self.providers = providers

        self.text_enc_sess = self._load_session("text_encoder.onnx")
        self.ref_enc_sess = self._load_session("reference_encoder.onnx", required=False)

        vf_name = "backbone_keys.onnx" if os.path.exists(os.path.join(self.config.onnx_dir, "backbone_keys.onnx")) else "backbone.onnx"
        self._vf_model_name = vf_name.replace(".onnx", "")
        self.vf_sess = self._load_session(vf_name)
        self.vocoder_sess = self._load_session("vocoder.onnx")
        self.dp_sess = self._load_session("length_pred.onnx", required=False)
        self.dp_style_sess = self._load_session("length_pred_style.onnx", required=False)
        
        if self.ref_enc_sess is None:
             print("Warning: Reference Encoder ONNX not found. You must provide style_json for inference.")
             
        # Cache VF input/output names
        self.vf_input_names = {i.name for i in self.vf_sess.get_inputs()}
        self.vf_supports_style_keys = "style_keys" in self.vf_input_names
        self.vf_has_style_mask = "style_mask" in self.vf_input_names
        self.vf_uses_text_emb = "text_emb" in self.vf_input_names and "text_context" not in self.vf_input_names

    def _load_session(self, name: str, required: bool = True) -> Optional[ort.InferenceSession]:
        use_slim = os.environ.get("USE_SLIM", "1") == "1"
        base = os.path.join(self.config.onnx_dir, name)
        if self.config.use_int8 and name.endswith(".onnx"):
            int8_name = name.replace(".onnx", "_int8.onnx")
            int8_base = os.path.join(self.config.onnx_dir, int8_name)
            if os.path.exists(int8_base):
                base = int8_base
        slim = base.replace(".onnx", ".slim.onnx")
        
        path = base
        if use_slim and os.path.exists(slim):
            print(f"  Using slim model: {os.path.basename(slim)}")
            path = slim
        elif not os.path.exists(base):
            if required:
                raise FileNotFoundError(f"Model {name} not found at {base}")
            return None
            
        return ort.InferenceSession(path, sess_options=self.sess_opts, providers=self.providers)

    def _load_stats(self):
        stats_path = os.path.join(self.config.onnx_dir, "stats.npz")
        self.mean = None
        self.std = None
        
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = stats['mean'].astype(np.float32)
            self.std = stats['std'].astype(np.float32)
            if self.mean.ndim == 1:
                self.mean = self.mean.reshape(1, -1, 1)
                self.std = self.std.reshape(1, -1, 1)
            if self.mean.ndim == 3:
                self.compressed_channels = int(self.mean.shape[1])
            print("Loaded stats from stats.npz")
            if 'normalizer_scale' in stats.files:
                ns = float(stats['normalizer_scale'].item() if stats['normalizer_scale'].ndim == 0 else stats['normalizer_scale'][0])
                if ns != self.normalizer_scale:
                    print(f"Warning: normalizer_scale in stats.npz ({ns}) does not match expected value ({self.normalizer_scale}). Using expected value.")
                    self.normalizer_scale = ns
            print(f"Loaded stats from stats.npz: mean={self.mean.shape}, std={self.std.shape}, normalizer_scale={self.normalizer_scale}")
        else:
            print("Warning: stats.npz not found. You may need to set .mean and .std manually.")

    def _load_uncond(self):
        uncond_path = os.path.join(self.config.onnx_dir, "uncond.npz")
        if os.path.exists(uncond_path):
            uncond = np.load(uncond_path)
            self.u_text = uncond['u_text']
            self.u_ref = uncond['u_ref']
            self.u_keys = uncond['u_keys'] if 'u_keys' in uncond.files else None
            self.cond_keys = uncond['cond_keys'] if 'cond_keys' in uncond.files else None
        else:
            print("Warning: uncond.npz not found. CFG will be disabled (or use dummy).")
            self.u_text = None
            self.u_ref = None
            self.u_keys = None
            self.cond_keys = None

    def _load_shuffle_keys(self):
        """Load extracted shuffle keys (keys.npz) for optimizer-proof obfuscation.

        When models are exported with key extraction, the inv_shuffle_indices
        arrays are stripped from the ONNX files and saved separately in keys.npz.
        This prevents ONNX Runtime's optimizer from constant-folding the Gather
        nodes, keeping the weight scrambling intact on disk.

        The keys must be fed as extra inputs at inference time.
        """
        self._model_keys = {}
        keys_path = os.path.join(self.config.onnx_dir, "keys.npz")
        if not os.path.exists(keys_path):
            return
        data = np.load(keys_path)
        for npz_key in data.files:
            parts = npz_key.split("/", 1)
            if len(parts) == 2:
                model_name, input_name = parts
                if model_name not in self._model_keys:
                    self._model_keys[model_name] = {}
                self._model_keys[model_name][input_name] = data[npz_key]
        total = sum(len(v) for v in self._model_keys.values())
        if total:
            print(f"Loaded {total} shuffle keys for {len(self._model_keys)} models from keys.npz")

    def _run(self, sess, feed, model_name):
        """Run ONNX session, injecting shuffle keys if available."""
        keys = self._model_keys.get(model_name)
        if keys:
            feed = {**feed, **keys}
        return sess.run(None, feed)

    def _phonemize(self, text: str) -> str:
        """Hebrew G2P pipeline: Text -> Vocalized (Phonikud) -> Phonemes (Phonikud).
        
        For long raw Hebrew text (>1024 chars), automatically chunks at 1024
        before running Phonikud to avoid memory/quality issues, then
        phonemizes each chunk and joins the results.
        """
        is_hebrew = any('\u0590' <= c <= '\u05ff' for c in text)
        has_nikud = any('\u05b0' <= c <= '\u05c7' for c in text)

        if not is_hebrew:
            print("[INFO] Input text appears to be phonemized (IPA).")
            return text
        elif has_nikud:
             print("[INFO] Input text has Nikud. Skipping Phonikud prediction.")
             if phonemize is None:
                 raise RuntimeError("phonikud not installed but input has Hebrew nikud.")
             return phonemize(text)
        else:
             print("[INFO] Input text is raw Hebrew. Running full Phonikud pipeline.")
             if self.phonikud is None or phonemize is None:
                 raise RuntimeError("phonikud not available for raw Hebrew text.")
             # Chunk long text at 1024 chars for Phonikud to avoid memory/quality issues
             if len(text) > 1024:
                 chunks = self._chunk_text(text, max_len=1024)
                 print(f"[INFO] Text length {len(text)} > 1024, splitting into {len(chunks)} chunks for Phonikud.")
                 phonemized_parts = []
                 for idx, chunk in enumerate(chunks):
                     print(f"[INFO] Phonikud chunk {idx+1}/{len(chunks)} ({len(chunk)} chars)")
                     vocalized = self.phonikud.add_diacritics(chunk)
                     phonemized_parts.append(phonemize(vocalized))
                 return " ".join(phonemized_parts)
             else:
                 vocalized = self.phonikud.add_diacritics(text)
                 return phonemize(vocalized)

    def prepare_text_input(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        phonemes = self._phonemize(text)
        print(f"Phonemes: {phonemes}")
        indices = text_to_indices(phonemes)
        text_ids = np.array([indices], dtype=np.int64)
        text_mask = np.ones((1, 1, len(indices)), dtype=np.float32)
        return text_ids, text_mask

    def _chunk_text(self, text: str, max_len: int = 150) -> List[str]:
        """
        Split text into chunks by paragraphs and sentences.

        Args:
            text: Input text to chunk
            max_len: Maximum length of each chunk (default: 150)

        Returns:
            List of text chunks
        """
        # Split by paragraph (two or more newlines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

        chunks = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Split by sentence boundaries (period, question mark, exclamation mark, comma followed by space)
            # But exclude common abbreviations like Mr., Mrs., Dr., etc. and single capital letters like F.
            pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?,\n])\s+"
            sentences = re.split(pattern, paragraph)

            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_len:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

        return chunks

    def load_style_json(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        with open(path, "r") as f:
            j = json.load(f)
        style_ttl = np.array(j["style_ttl"]["data"], dtype=np.float32) if "style_ttl" in j else None
        style_keys = np.array(j["style_keys"]["data"], dtype=np.float32) if "style_keys" in j else None
        style_dp = np.array(j["style_dp"]["data"], dtype=np.float32) if "style_dp" in j else None
        z_ref = None
        if "z_ref" in j:
            z_ref = np.array(j["z_ref"]["data"], dtype=np.float32)
            if z_ref.ndim == 2:
                z_ref = z_ref[None, :, :]
        if style_ttl is not None and style_ttl.ndim == 2:
            style_ttl = style_ttl[None, :, :]
        if style_keys is not None and style_keys.ndim == 2:
            style_keys = style_keys[None, :, :]
        if style_dp is not None and style_dp.ndim == 2:
            style_dp = style_dp[None, :, :]
        return style_ttl, style_keys, style_dp, z_ref

    def extract_style(self, z_ref_norm: np.ndarray, out_json: str = None) -> Dict[str, Any]:
        if self.ref_enc_sess is None:
            raise ValueError("Reference Encoder not loaded.")

        z_ref_in = z_ref_norm.astype(np.float32)

        TARGET_REF_LEN = 256
        B, C, T = z_ref_in.shape
        if T < TARGET_REF_LEN:
            pad_amt = TARGET_REF_LEN - T
            z_ref_padded = np.pad(z_ref_in, ((0, 0), (0, 0), (0, pad_amt)), mode="constant")
            ref_mask = np.zeros((B, 1, TARGET_REF_LEN), dtype=np.float32)
            ref_mask[:, :, :T] = 1.0
        elif T > TARGET_REF_LEN:
            z_ref_padded = z_ref_in[:, :, :TARGET_REF_LEN]
            ref_mask = np.ones((B, 1, TARGET_REF_LEN), dtype=np.float32)
        else:
            z_ref_padded = z_ref_in
            ref_mask = np.ones((B, 1, TARGET_REF_LEN), dtype=np.float32)

        ref_input_names = [i.name for i in self.ref_enc_sess.get_inputs()]
        ref_feed = {"z_ref": z_ref_padded}
        if "mask" in ref_input_names:
            ref_feed["mask"] = ref_mask
        elif "ref_mask" in ref_input_names:
            ref_feed["ref_mask"] = ref_mask
        else:
            if len(ref_input_names) >= 2:
                ref_feed[ref_input_names[1]] = ref_mask

        ref_values, ref_keys = self._run(self.ref_enc_sess, ref_feed, "reference_encoder")[:2]

        payload = {
            "style_ttl": {"data": ref_values.tolist(), "dims": list(ref_values.shape)},
            "style_keys": {"data": ref_keys.tolist(), "dims": list(ref_keys.shape)},
        }

        if out_json:
            with open(out_json, "w") as f:
                json.dump(payload, f)
            print(f"[OK] Saved style JSON: {out_json}")

        return payload

    def infer_stream(self, text: str, z_ref: Optional[np.ndarray] = None, style_json_path: Optional[str] = None, **kwargs):
        """
        Generator that yields audio chunks.
        
        Implements two-stage chunking for optimal quality:
        1. Large chunks (max 1024) for Phonikud G2P with proper context
        2. Small chunks (max text_chunk_len, default 150) for TTS inference
        
        This ensures the model ALWAYS works with appropriately-sized chunks,
        matching the reference implementation's best practices.
        """
        # Apply overrides
        steps = kwargs.get('steps', self.config.steps)
        cfg_scale = kwargs.get('cfg_scale', self.config.cfg_scale)
        speed = kwargs.get('speed', self.config.speed)
        seed = kwargs.get('seed', self.config.seed)
        
        # ALWAYS chunk input text - model works best with chunks
        # Stage 1: Large chunks for Phonikud G2P (max 1024 for context)
        large_chunks = self._chunk_text(text, max_len=1024)
        print(f"[CHUNKING] Split input ({len(text)} chars) into {len(large_chunks)} large chunks (max 1024).")
        
        silence_samples = int(self.config.silence_sec * self.sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)
        
        total_small_chunks = 0
        
        for i, large_chunk in enumerate(large_chunks):
            # Apply Phonikud (add diacritics) if needed for raw Hebrew text
            is_hebrew = any('\u0590' <= c <= '\u05ff' for c in large_chunk)
            has_nikud = any('\u05b0' <= c <= '\u05c7' for c in large_chunk)
            
            processed_text = large_chunk
            if is_hebrew and not has_nikud and self.phonikud is not None:
                 print(f"[PHONIKUD] Processing large chunk {i+1}/{len(large_chunks)} ({len(large_chunk)} chars)")
                 try:
                     processed_text = self.phonikud.add_diacritics(large_chunk)
                 except Exception as e:
                     print(f"[WARN] Phonikud failed: {e}. Using raw text.")
            
            # Stage 2: ALWAYS split into small chunks for TTS inference
            # Model performs best with chunks around text_chunk_len (default 150)
            small_chunks = self._chunk_text(processed_text, max_len=int(self.config.text_chunk_len))
            print(f"[CHUNKING] Large chunk {i+1} split into {len(small_chunks)} TTS chunks (max {self.config.text_chunk_len}).")
            
            for j, small_chunk in enumerate(small_chunks):
                chunk_id = total_small_chunks + j + 1
                print(f"[TTS] Processing chunk {chunk_id} (large {i+1}.{j+1}): {len(small_chunk)} chars")
                w = self._infer_single(small_chunk, z_ref, style_json_path, steps, cfg_scale, speed, seed)
                
                yield w
                
                # Add silence between chunks (except after the very last one)
                if i < len(large_chunks) - 1 or j < len(small_chunks) - 1:
                    yield silence
            
            total_small_chunks += len(small_chunks)
        
        print(f"[DONE] Processed {total_small_chunks} total TTS chunks.")

    def infer(self, text: str, z_ref: Optional[np.ndarray] = None, style_json_path: Optional[str] = None, stream: bool = False, **kwargs):
        """
        Main inference entry point.
        kwargs can override config values (steps, cfg_scale, speed, seed).
        If stream=True, returns a generator that yields audio chunks.
        If stream=False, returns the full concatenated audio array.
        """
        gen = self.infer_stream(text, z_ref, style_json_path, **kwargs)
        
        if stream:
            return gen

        wav_chunks = list(gen)
        if not wav_chunks:
             return np.array([], dtype=np.float32)
             
        return np.concatenate(wav_chunks)

    def _infer_single(self, text: str, z_ref, style_json_path, steps, cfg_scale, speed, seed) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Stats not initialized.")
            
        # 1. Load Style/Z_ref from JSON first if available
        style_ttl, style_keys, style_dp, style_dp_input = None, None, None, None
        
        if style_json_path:
            style_ttl, style_keys, style_dp, json_z_ref = self.load_style_json(style_json_path)
            if z_ref is None and json_z_ref is not None:
                z_ref = json_z_ref
        
        if z_ref is None and style_ttl is None:
            raise ValueError("Must provide either z_ref or style_json_path (with valid content)")

        # 2. Text
        text_ids, text_mask = self.prepare_text_input(text)
        
        # 3. Style
        z_ref_norm = None
        if z_ref is not None:
            z_ref_norm = ((z_ref - self.mean) / self.std) * float(self.normalizer_scale)

            if z_ref_norm is not None and z_ref_norm.ndim == 3:
                T = int(z_ref_norm.shape[2])
                # Trim tail frames (last ~5%) to remove boundary artifacts
                # (matches reference implementation's build_reference_runtime)
                tail_trim = max(2, int(T * 0.05))
                T_trimmed = max(1, T - tail_trim)
                if T_trimmed < T:
                    z_ref_norm = z_ref_norm[:, :, :T_trimmed]
                    T = T_trimmed
                
                # Cap reference to target_frames (default 150, matching reference script)
                target_frames = int(self.config.ref_target_frames)
                if T > target_frames:
                    print(f"[INFO] Capping reference from {T} to {target_frames} frames.")
                    z_ref_norm = z_ref_norm[:, :, :target_frames]

        if style_ttl is not None:
            ref_values = style_ttl
            if style_dp is not None:
                style_dp_input = style_dp
        else:
            payload = self.extract_style(z_ref_norm)
            ref_values = np.array(payload["style_ttl"]["data"], dtype=np.float32)
            style_keys = np.array(payload["style_keys"]["data"], dtype=np.float32) if "style_keys" in payload else None

        # Ensure batch dim for style arrays
        if ref_values is not None and ref_values.ndim == 2:
            ref_values = ref_values[None, :, :]
        if style_keys is not None and style_keys.ndim == 2:
            style_keys = style_keys[None, :, :]

        # TextEncoder ref_keys: prefer explicit style_keys, else fall back to ref_values
        ref_keys = style_keys if style_keys is not None else ref_values

        # 4. Text encoder
        text_input_names = {i.name for i in self.text_enc_sess.get_inputs()}
        te_feed = {"text_ids": text_ids}
        if "text_mask" in text_input_names:
            te_feed["text_mask"] = text_mask
        if "style_ttl" in text_input_names:
            te_feed["style_ttl"] = ref_values
        elif "ref_values" in text_input_names:
            te_feed["ref_values"] = ref_values
        else:
            raise ValueError()

        # Some exports require ref_keys (even if style_ttl is present)
        if "ref_keys" in text_input_names:
            te_feed["ref_keys"] = ref_keys
        elif "used_ref_keys" in text_input_names:
            te_feed["used_ref_keys"] = ref_keys

        text_emb = self._run(self.text_enc_sess, te_feed, "text_encoder")[0]

        # 5. Duration
        z_ref_dp = z_ref_norm
        T_lat = self._predict_duration(text_ids, text_mask, z_ref_dp, style_dp_input, speed)

        # 6. Flow matching
        x = self._sample_flow_matching(text_emb, ref_values, text_mask, T_lat, steps, cfg_scale, seed)

        # 7. Vocoder
        wav = self._decode_waveform(x)
        return wav

    def _predict_duration(self, text_ids, text_mask, z_ref_dp, style_dp_input, speed):
        T_lat = None

        if style_dp_input is not None and self.dp_style_sess is not None:
            dur_out = self._run(self.dp_style_sess, {"text_ids": text_ids, "style_dp": style_dp_input, "text_mask": text_mask}, "length_pred_style")
            raw_val = float(np.squeeze(dur_out[0]))
            if np.isfinite(raw_val):
                T_lat = int(np.round(raw_val / max(speed, 1e-6)))

        if T_lat is None and z_ref_dp is not None and self.dp_sess is not None:
            ref_len = int(z_ref_dp.shape[2])
            dur_out = self._run(
                self.dp_sess,
                {
                    "text_ids": text_ids,
                    "z_ref": z_ref_dp.astype(np.float32),
                    "text_mask": text_mask,
                    "ref_mask": np.ones((1, 1, ref_len), dtype=np.float32),
                },
                "length_pred",
            )
            raw_val = float(np.squeeze(dur_out[0]))
            if np.isfinite(raw_val):
                T_lat = int(np.round(raw_val / max(speed, 1e-6)))

        if T_lat is None:
            T_lat = int(text_ids.shape[1] * 1.3)

        txt_len = int(np.sum(text_mask))
        T_cap = int(max(20, min(txt_len * 3 + 20, 600)))
        T_lat = min(max(int(T_lat), 1), T_cap)
        T_lat = min(T_lat, 800)
        return max(10, T_lat)

    def _sample_flow_matching(self, text_emb, ref_values, text_mask, T_lat, steps, cfg_scale, seed):
        # Use a local random generator to avoid affecting global state
        rng = np.random.RandomState(seed)
        x = rng.randn(1, self.compressed_channels, T_lat).astype(np.float32)
        latent_mask = np.ones((1, 1, T_lat), dtype=np.float32)
        if cfg_scale != 1.0:
            if not self.vf_supports_style_keys:
                raise RuntimeError()
            if self.u_text is None or self.u_ref is None or self.u_keys is None:
                raise RuntimeError()
            if self.cond_keys is None:
                raise RuntimeError()

        vf_style_keys_cond = None
        if self.vf_supports_style_keys and self.cond_keys is not None:
            vf_style_keys_cond = self.cond_keys.astype(np.float32)
            if vf_style_keys_cond.ndim == 2:
                vf_style_keys_cond = vf_style_keys_cond[None, :, :]

        u_text_ctx = self.u_text.astype(np.float32) if self.u_text is not None else None
        u_ref = self.u_ref.astype(np.float32) if self.u_ref is not None else None
        u_keys = self.u_keys.astype(np.float32) if self.u_keys is not None else None
        u_text_mask = np.ones((1, 1, 1), dtype=np.float32)

        vf_input_names = self.vf_input_names
        for i in range(int(steps)):
            t_val = np.array([float(i)], dtype=np.float32)
            total_t = np.array([float(steps)], dtype=np.float32)

            vf_feed = {}
            if "noisy_latent" in vf_input_names:
                vf_feed["noisy_latent"] = x
            if "text_emb" in vf_input_names:
                vf_feed["text_emb"] = text_emb
            elif "text_context" in vf_input_names:
                vf_feed["text_context"] = text_emb
            if "style_ttl" in vf_input_names:
                vf_feed["style_ttl"] = ref_values
            elif "ref_values" in vf_input_names:
                vf_feed["ref_values"] = ref_values
            if "latent_mask" in vf_input_names:
                vf_feed["latent_mask"] = latent_mask
            if "text_mask" in vf_input_names:
                vf_feed["text_mask"] = text_mask
            if "current_step" in vf_input_names:
                vf_feed["current_step"] = t_val
            if "total_step" in vf_input_names:
                vf_feed["total_step"] = total_t
            if "style_keys" in vf_input_names and vf_style_keys_cond is not None:
                vf_feed["style_keys"] = vf_style_keys_cond
            if "style_mask" in vf_input_names:
                vf_feed["style_mask"] = np.ones((1, 1, ref_values.shape[1]), dtype=np.float32)

            den_cond = self._run(self.vf_sess, vf_feed, self._vf_model_name)[0]

            if cfg_scale != 1.0:
                vf_feed_u = dict(vf_feed)
                if "text_emb" in vf_input_names:
                    vf_feed_u["text_emb"] = u_text_ctx
                elif "text_context" in vf_input_names:
                    vf_feed_u["text_context"] = u_text_ctx
                if "style_ttl" in vf_input_names:
                    vf_feed_u["style_ttl"] = u_ref
                elif "ref_values" in vf_input_names:
                    vf_feed_u["ref_values"] = u_ref
                if "text_mask" in vf_input_names:
                    vf_feed_u["text_mask"] = u_text_mask
                if "style_keys" in vf_input_names:
                    vf_feed_u["style_keys"] = u_keys
                if "style_mask" in vf_input_names:
                    vf_feed_u["style_mask"] = np.ones((1, 1, u_ref.shape[1]), dtype=np.float32)

                den_uncond = self._run(self.vf_sess, vf_feed_u, self._vf_model_name)[0]
                x = den_uncond + cfg_scale * (den_cond - den_uncond)
            else:
                x = den_cond

        return x

    def _apply_fade(self, wav: np.ndarray) -> np.ndarray:
        """Apply fade in/out to prevent clicks at audio boundaries"""
        if self.config.fade_duration <= 0:
            return wav

        fade_samples = int(self.config.fade_duration * self.sample_rate)
        if fade_samples == 0 or len(wav) < 2 * fade_samples:
            return wav

        # Make a copy to avoid modifying the input
        wav = wav.copy()

        # Create fade curves
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        # Apply fades
        wav[:fade_samples] *= fade_in
        wav[-fade_samples:] *= fade_out

        return wav

    def _decode_waveform(self, z_pred):
        if self.mean is None or self.std is None:
            raise ValueError()

        if float(self.normalizer_scale) not in (0.0, 1.0):
            z_pred_unnorm = (z_pred / float(self.normalizer_scale)) * self.std + self.mean
        else:
            z_pred_unnorm = z_pred * self.std + self.mean

        B, C, T = z_pred_unnorm.shape
        z_dec_in = (
            z_pred_unnorm.reshape(B, self.latent_dim, self.chunk_compress_factor, T)
            .transpose(0, 1, 3, 2)
            .reshape(B, self.latent_dim, T * self.chunk_compress_factor)
        )

        wav = self._run(self.vocoder_sess, {"latent": z_dec_in}, "vocoder")[0]

        frame_len = int(self.hop_length * self.chunk_compress_factor)
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]

        wav = wav.squeeze()

        # Apply fade to prevent clicks
        if self.config.fade_duration > 0:
            wav = self._apply_fade(wav)

        return wav

if __name__ == "__main__":
    import soundfile as sf
    # import torch # Removed global import to allow running without torch if not using .pt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="onnx_models")
    parser.add_argument("--phonikud_path", default="phonikud-1.0.onnx")
    parser.add_argument("--text", default="שלום עולם")
    parser.add_argument("--z_ref", default=None)
    parser.add_argument("--style_json", default="voices/male1.json")
    parser.add_argument("--out", default="out.wav")
    
    # Settings handled by Config
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = TTSConfig(
        onnx_dir=args.onnx_dir,
        phonikud_path=args.phonikud_path,
        steps=args.steps,
        cfg_scale=args.cfg,
        speed=args.speed,
        seed=args.seed,
    )
    
    tts = HebrewTTS(config)
    
    # Load z_ref if needed
    z_ref = None
    if args.z_ref:
        if args.z_ref.endswith(".pt"):
            import torch
            z_ref_loaded = torch.load(args.z_ref, map_location='cpu')
            if isinstance(z_ref_loaded, dict):
                 # Try common keys
                 for k in ['z_ref_raw', 'z', 'latent']:
                     if k in z_ref_loaded:
                         z_ref = z_ref_loaded[k]
                         break
                 if z_ref is None:
                     # fallback first tensor
                     for v in z_ref_loaded.values():
                         if torch.is_tensor(v):
                             z_ref = v
                             break
            else:
                z_ref = z_ref_loaded
            if torch.is_tensor(z_ref): z_ref = z_ref.numpy()
        else:
            z_ref = np.load(args.z_ref)
        if z_ref is not None and z_ref.ndim == 2: z_ref = z_ref[None, :, :]
            
    if z_ref is None and args.style_json is None:
        print("Error: Must provide --z_ref or --style_json")
        sys.exit(1)
        
    t0 = time.time()
    # infer uses config defaults unless overridden
    wav = tts.infer(args.text, z_ref=z_ref, style_json_path=args.style_json)
    t1 = time.time()
    
    dur = len(wav) / float(tts.sample_rate) if len(wav) > 0 else 0.0
    print(f"Generated {dur:.2f}s in {t1-t0:.2f}s (RTF: {(t1-t0)/dur:.3f})" if dur > 0 else f"Generated {dur:.2f}s in {t1-t0:.2f}s")
    sf.write(args.out, wav, tts.sample_rate)
    print(f"Saved {args.out}")
