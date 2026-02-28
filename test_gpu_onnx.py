"""
test_gpu_onnx.py — GPU ONNX benchmark for Hebrew TTS

Tests the FULL pipeline — every ONNX model on GPU:
  Raw Hebrew text → Phonikud (GPU) → text_encoder → lenght_pred
    → flow-matching / backbone (N steps) → vocoder → WAV

PhoniKud's internal ORT session is also replaced with the GPU provider
(TensorRT or CUDA) so 100 % of inference runs on the GPU.

Providers:
  tensorrt  — TensorrtExecutionProvider → CUDAExecutionProvider → CPU
  cuda      — CUDAExecutionProvider → CPU
  cpu       — CPUExecutionProvider only
  auto      — best available (TRT > CUDA > CPU)

A warm-up pass is always performed before timed runs to allow TensorRT
engine compilation and CUDA kernel caching.

Usage (run from hebrew/ directory):
    python test_gpu_onnx.py                                   # auto provider, 1 run
    python test_gpu_onnx.py --provider tensorrt --fp16        # TRT FP16
    python test_gpu_onnx.py --provider cuda                   # CUDA only
    python test_gpu_onnx.py --runs 3                          # 3 timed runs + summary
    python test_gpu_onnx.py --compare                         # TRT vs CUDA side-by-side
    python test_gpu_onnx.py --no_warmup                       # skip the warm-up pass
"""

import os
import sys
import time
import json
import argparse
import textwrap
from typing import Optional, List, Dict, Any

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Resolve script location so relative paths work wherever you run from
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Defaults  (edit or override with CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_TEXT       = "שלום, זהו ניסיון בדיקה של מערכת יצירת הדיבור בעברית."
DEFAULT_VOICE      = "voices/female1.json"
DEFAULT_STEPS      = 5
DEFAULT_CFG        = 3.0
DEFAULT_SPEED      = 1.0
DEFAULT_SEED       = 42
DEFAULT_ONNX_DIR   = "onnx_models"
DEFAULT_PHONIKUD   = "phonikud-1.0.onnx"
DEFAULT_PROVIDER   = "auto"   # auto | tensorrt | cuda | cpu
DEFAULT_RUNS       = 1
DEFAULT_OUT_WAV    = "gpu_test_output.wav"
DEFAULT_FP16       = False     # TensorRT FP16 flag
DEFAULT_COMPARE    = False     # run TRT vs CUDA side-by-side
DEFAULT_WARMUP     = True      # always warm up before timed runs

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class TimedSession:
    """Wraps an ort.InferenceSession and records cumulative call stats."""

    def __init__(self, session: ort.InferenceSession, name: str):
        self._sess   = session
        self.name    = name
        self.calls   = 0
        self.total_s = 0.0
        self.timings: List[float] = []

    # Forward attribute access (get_inputs, get_outputs, etc.)
    def __getattr__(self, item):
        return getattr(self._sess, item)

    def run(self, output_names, input_feed, run_options=None):
        t0 = time.perf_counter()
        out = self._sess.run(output_names, input_feed, run_options)
        elapsed = time.perf_counter() - t0
        self.calls   += 1
        self.total_s += elapsed
        self.timings.append(elapsed)
        return out

    def reset(self):
        self.calls   = 0
        self.total_s = 0.0
        self.timings.clear()

    def summary(self) -> str:
        if self.calls == 0:
            return f"  {self.name:<35} — not called"
        avg = self.total_s / self.calls
        mn  = min(self.timings)
        mx  = max(self.timings)
        return (
            f"  {self.name:<35}  calls={self.calls:>4}  "
            f"total={self.total_s*1000:>8.1f} ms  "
            f"avg={avg*1000:>7.1f} ms  "
            f"min={mn*1000:>7.1f} ms  "
            f"max={mx*1000:>7.1f} ms"
        )


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------

def _trt_provider_options(fp16: bool = False) -> Dict[str, Any]:
    return {
        "device_id": 0,
        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4 GiB
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": os.path.join(SCRIPT_DIR, ".trt_cache"),
    }


def _cuda_provider_options() -> Dict[str, Any]:
    return {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kNextPowerOfTwo",
    }


def build_providers(preferred: str, fp16: bool) -> list:
    """Return an ordered provider list based on the user's preference and GPU availability."""
    available = ort.get_available_providers()
    print(f"[ORT] Available providers: {available}")

    has_trt  = "TensorrtExecutionProvider"  in available
    has_cuda = "CUDAExecutionProvider"      in available

    preferred = preferred.lower()

    if preferred == "tensorrt":
        if has_trt:
            return [
                ("TensorrtExecutionProvider",  _trt_provider_options(fp16)),
                ("CUDAExecutionProvider",       _cuda_provider_options()),
                "CPUExecutionProvider",
            ]
        print("[WARN] TensorRT not available — falling back to CUDA.")
        preferred = "cuda"

    if preferred == "cuda":
        if has_cuda:
            return [
                ("CUDAExecutionProvider", _cuda_provider_options()),
                "CPUExecutionProvider",
            ]
        print("[WARN] CUDA not available — falling back to CPU.")
        return ["CPUExecutionProvider"]

    if preferred == "auto":
        providers: list = []
        if has_trt:
            providers.append(("TensorrtExecutionProvider", _trt_provider_options(fp16)))
        if has_cuda:
            providers.append(("CUDAExecutionProvider", _cuda_provider_options()))
        providers.append("CPUExecutionProvider")
        return providers

    # cpu
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Phonikud GPU session injection helper
# ---------------------------------------------------------------------------

def _inject_gpu_into_phonikud(phonikud_obj, phonikud_path: str, providers: list, opts: ort.SessionOptions):
    """
    Replace the internal ORT InferenceSession inside a phonikud_onnx.Phonikud
    object with one loaded under the GPU providers we chose.

    phonikud_onnx stores its session under one of several attribute names
    depending on the library version.  We probe the most common ones.
    """
    candidate_attrs = ["session", "model", "sess", "ort_session", "_session", "_model"]
    replaced = False
    for attr in candidate_attrs:
        inner = getattr(phonikud_obj, attr, None)
        if inner is not None and isinstance(inner, ort.InferenceSession):
            try:
                gpu_sess = ort.InferenceSession(phonikud_path, sess_options=opts, providers=providers)
                setattr(phonikud_obj, attr, gpu_sess)
                print(f"[PHONIKUD] Replaced internal session (attr='{attr}') with GPU providers.")
                replaced = True
            except Exception as e:
                print(f"[PHONIKUD][WARN] Could not load phonikud on GPU (attr='{attr}'): {e}")
            break

    if not replaced:
        # Fallback: try loading a fresh session and swapping any InferenceSession we find
        for attr in dir(phonikud_obj):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(phonikud_obj, attr)
            except Exception:
                continue
            if isinstance(val, ort.InferenceSession):
                try:
                    gpu_sess = ort.InferenceSession(phonikud_path, sess_options=opts, providers=providers)
                    setattr(phonikud_obj, attr, gpu_sess)
                    print(f"[PHONIKUD] Replaced internal session (attr='{attr}') with GPU providers (fallback scan).")
                    replaced = True
                except Exception as e:
                    print(f"[PHONIKUD][WARN] Fallback GPU swap failed: {e}")
                break

    if not replaced:
        print("[PHONIKUD][WARN] Could not find ORT session inside Phonikud object — running Phonikud on CPU.")


# ---------------------------------------------------------------------------
# Instrumented subclass of HebrewTTS
# ---------------------------------------------------------------------------

from hebrew_inference_helper import HebrewTTS, TTSConfig

class InstrumentedHebrewTTS(HebrewTTS):
    """HebrewTTS with timing wrappers around every ONNX session."""

    def __init__(self, config: TTSConfig, providers: list):
        self._override_providers = providers
        # We'll collect per-stage phase timings separately
        self.phase_timings: Dict[str, float] = {}
        super().__init__(config)
        # Inject GPU providers into Phonikud's internal session
        if self.phonikud is not None:
            _inject_gpu_into_phonikud(
                self.phonikud,
                config.phonikud_path,
                self._override_providers,
                self.sess_opts,
            )
        self._wrap_sessions()

    # ------------------------------------------------------------------
    # Override _init_onnx_sessions so we can inject custom providers
    # ------------------------------------------------------------------
    def _init_onnx_sessions(self):
        print(f"\n[ONNX] Loading sessions with providers = {self._override_providers}")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        intra_def = os.environ.get("ORT_INTRA")
        inter_def = os.environ.get("ORT_INTER", "1")
        cpu_cores = max(1, (os.cpu_count() or 2) // 4)
        opts.intra_op_num_threads = int(intra_def) if intra_def else cpu_cores
        opts.inter_op_num_threads = int(inter_def)
        print(f"[ORT] Threads: intra={opts.intra_op_num_threads}, inter={opts.inter_op_num_threads}")

        self.sess_opts = opts
        self.providers = self._override_providers

        self.text_enc_sess  = self._load_session("text_encoder.onnx")
        self.ref_enc_sess   = self._load_session("reference_encoder.onnx", required=False)

        vf_name = (
            "backbone_keys.onnx"
            if os.path.exists(os.path.join(self.config.onnx_dir, "backbone_keys.onnx"))
            else "backbone.onnx"
        )
        self.vf_sess      = self._load_session(vf_name)
        self.vocoder_sess = self._load_session("vocoder.onnx")
        self.dp_sess      = self._load_session("lenght_pred.onnx",       required=False)
        self.dp_style_sess= self._load_session("lenght_predictor_style.onnx", required=False)

        if self.ref_enc_sess is None:
            print("[WARN] Reference Encoder not found — must supply style_json_path.")

        self.vf_input_names       = {i.name for i in self.vf_sess.get_inputs()}
        self.vf_supports_style_keys = "style_keys"    in self.vf_input_names
        self.vf_has_style_mask    = "style_mask"      in self.vf_input_names
        self.vf_uses_text_emb     = "text_emb"        in self.vf_input_names and "text_context" not in self.vf_input_names

    # ------------------------------------------------------------------
    # Wrap each session in a TimedSession after loading
    # ------------------------------------------------------------------
    def _wrap_sessions(self):
        self.timed_sessions: Dict[str, TimedSession] = {}

        def wrap(attr: str, label: str):
            sess = getattr(self, attr, None)
            if sess is not None and not isinstance(sess, TimedSession):
                ts = TimedSession(sess, label)
                setattr(self, attr, ts)
                self.timed_sessions[label] = ts

        wrap("text_enc_sess",   "text_encoder")
        wrap("ref_enc_sess",    "reference_encoder")
        wrap("vf_sess",         "backbone (flow)")
        wrap("vocoder_sess",    "vocoder")
        wrap("dp_sess",         "lenght_pred")
        wrap("dp_style_sess",   "lenght_predictor_style")

    def reset_timings(self):
        for ts in self.timed_sessions.values():
            ts.reset()
        self.phase_timings.clear()


# ---------------------------------------------------------------------------
# Phonikud timing wrapper
# ---------------------------------------------------------------------------

class TimedPhonikud:
    """Wraps Phonikud.add_diacritics() with timing."""

    def __init__(self, inner):
        self._inner   = inner
        self.calls    = 0
        self.total_s  = 0.0
        self.timings: List[float] = []

    def add_diacritics(self, text: str) -> str:
        t0 = time.perf_counter()
        result = self._inner.add_diacritics(text)
        elapsed = time.perf_counter() - t0
        self.calls   += 1
        self.total_s += elapsed
        self.timings.append(elapsed)
        return result

    # Forward everything else (e.g. phonikud ONNX session access)
    def __getattr__(self, item):
        return getattr(self._inner, item)

    def reset(self):
        self.calls   = 0
        self.total_s = 0.0
        self.timings.clear()

    def summary(self) -> str:
        if self.calls == 0:
            return "  phonikud                             — not called"
        avg = self.total_s / self.calls
        mn  = min(self.timings)
        mx  = max(self.timings)
        return (
            f"  {'phonikud (add_diacritics)':<35}  calls={self.calls:>4}  "
            f"total={self.total_s*1000:>8.1f} ms  "
            f"avg={avg*1000:>7.1f} ms  "
            f"min={mn*1000:>7.1f} ms  "
            f"max={mx*1000:>7.1f} ms"
        )


# ---------------------------------------------------------------------------
# Pipeline-level timed runner
# ---------------------------------------------------------------------------

def run_pipeline(
    tts: InstrumentedHebrewTTS,
    text: str,
    voice_path: str,
    steps: int,
    cfg: float,
    speed: float,
    seed: int,
) -> tuple:
    """
    Run full pipeline and return (wav_array, phase_times_dict, timed_phonikud).
    phase_times_dict keys: phonemization, text_encode_total, duration_total,
                           flow_total, vocoder_total, total
    """
    phases: Dict[str, float] = {}

    # ── 1. Phonemization (Phonikud + phonemize) ─────────────────────────────
    timed_ph = None
    if tts.phonikud is not None and not isinstance(tts.phonikud, TimedPhonikud):
        timed_ph = TimedPhonikud(tts.phonikud)
        tts.phonikud = timed_ph
    elif isinstance(tts.phonikud, TimedPhonikud):
        timed_ph = tts.phonikud

    t0 = time.perf_counter()
    # We exercise _phonemize indirectly through infer(); capture separately.

    # ── 2. Full infer call ───────────────────────────────────────────────────
    t_infer_start = time.perf_counter()
    wav = tts.infer(
        text,
        style_json_path=voice_path,
        steps=steps,
        cfg_scale=cfg,
        speed=speed,
        seed=seed,
    )
    t_total = time.perf_counter() - t_infer_start

    # ── 3. Collect per-session timing ────────────────────────────────────────
    session_times: Dict[str, float] = {}
    for label, ts in tts.timed_sessions.items():
        session_times[label] = ts.total_s

    phonikud_time = timed_ph.total_s if timed_ph else 0.0

    return wav, phonikud_time, session_times, t_total


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(
    text: str,
    providers: list,
    wav: np.ndarray,
    sample_rate: int,
    phonikud_time: float,
    session_times: Dict[str, float],
    t_total: float,
    steps: int,
    run_idx: int,
    runs: int,
    tts: InstrumentedHebrewTTS,
):
    audio_dur = len(wav) / sample_rate if len(wav) > 0 else 0.0
    rtf = t_total / audio_dur if audio_dur > 0 else float("inf")

    sep = "─" * 90

    print(f"\n{sep}")
    print(f"  RUN {run_idx}/{runs}  |  providers: {[p if isinstance(p, str) else p[0] for p in providers]}")
    print(sep)
    print(f"  Text ({len(text)} chars): {textwrap.shorten(text, 70)}")
    print(f"  Steps: {steps}   Audio duration: {audio_dur:.3f}s   RTF: {rtf:.4f}   Total wall: {t_total*1000:.1f} ms")
    print(sep)
    print("  STAGE BREAKDOWN")
    print(sep)

    # Reconstruct ordered timings
    ordered = [
        ("phonikud (add_diacritics)",    phonikud_time),
    ]
    for label, ts in tts.timed_sessions.items():
        if ts.calls > 0:
            ordered.append((label, ts.total_s))

    accounted = sum(v for _, v in ordered)
    other     = max(0.0, t_total - accounted)

    # Header
    print(f"  {'Stage':<35}  {'Time (ms)':>10}  {'% of total':>10}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")

    for label, t in ordered:
        pct = t / t_total * 100 if t_total > 0 else 0
        print(f"  {label:<35}  {t*1000:>10.1f}  {pct:>9.1f}%")

    print(f"  {'other (chunking, I/O, fade…)':<35}  {other*1000:>10.1f}  {other/t_total*100 if t_total>0 else 0:>9.1f}%")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
    print(f"  {'TOTAL':<35}  {t_total*1000:>10.1f}  {'100.0%':>10}")
    print(sep)

    # Per-session detailed stats
    print("  PER-SESSION CALL STATS")
    print(sep)
    for ts in tts.timed_sessions.values():
        print(ts.summary())

    if isinstance(tts.phonikud, TimedPhonikud):
        print(tts.phonikud.summary())
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU ONNX benchmark — Hebrew TTS full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text",     default=DEFAULT_TEXT,     help="Hebrew input text")
    parser.add_argument("--voice",    default=DEFAULT_VOICE,    help="Path to voice JSON (relative to script dir)")
    parser.add_argument("--steps",    type=int,   default=DEFAULT_STEPS,    help="Diffusion steps")
    parser.add_argument("--cfg",      type=float, default=DEFAULT_CFG,      help="CFG scale")
    parser.add_argument("--speed",    type=float, default=DEFAULT_SPEED,    help="Speed multiplier")
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED,     help="Random seed")
    parser.add_argument("--onnx_dir", default=DEFAULT_ONNX_DIR, help="ONNX models directory")
    parser.add_argument("--phonikud", default=DEFAULT_PHONIKUD, help="Phonikud ONNX path")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=["auto", "tensorrt", "cuda", "cpu"],
                        help="Execution provider preference")
    parser.add_argument("--fp16",      action="store_true", default=DEFAULT_FP16,
                        help="Enable FP16 for TensorRT")
    parser.add_argument("--runs",      type=int, default=DEFAULT_RUNS,
                        help="Number of inference runs (for averaging)")
    parser.add_argument("--compare",   action="store_true", default=DEFAULT_COMPARE,
                        help="Run TensorRT vs CUDA side-by-side and print comparison table")
    parser.add_argument("--no_warmup", action="store_true", default=False,
                        help="Skip the warm-up pass (not recommended for TensorRT)")
    parser.add_argument("--out",       default=DEFAULT_OUT_WAV, help="Output WAV file path")
    parser.add_argument("--no_save",   action="store_true",     help="Skip saving the WAV file")
    args = parser.parse_args()
    args.warmup = not args.no_warmup

    # Resolve paths relative to script dir
    onnx_dir_abs  = os.path.join(SCRIPT_DIR, args.onnx_dir)
    phonikud_abs  = os.path.join(SCRIPT_DIR, args.phonikud)
    voice_abs     = os.path.join(SCRIPT_DIR, args.voice)

    # ── Validate ────────────────────────────────────────────────────────────
    for path, label in [
        (onnx_dir_abs, "onnx_dir"),
        (phonikud_abs, "phonikud"),
        (voice_abs,    "voice"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)

    # ── Compare mode: TensorRT vs CUDA ───────────────────────────────────────
    if args.compare:
        _run_comparison(args, onnx_dir_abs, phonikud_abs, voice_abs)
        return

    # ── Build providers ─────────────────────────────────────────────────────
    providers = build_providers(args.provider, args.fp16)
    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    print(f"\n[INFO] Requested provider: '{args.provider}'  →  using: {provider_names}")

    # ── Build config ─────────────────────────────────────────────────────────
    config = TTSConfig(
        onnx_dir     = onnx_dir_abs,
        phonikud_path= phonikud_abs,
        use_gpu      = any("CUDA" in p or "Tensorrt" in p.capitalize() for p in provider_names),
        use_int8     = False,
        steps        = args.steps,
        cfg_scale    = args.cfg,
        speed        = args.speed,
        seed         = args.seed,
    )

    # ── Instantiate ──────────────────────────────────────────────────────────
    print("\n[INIT] Loading models…")
    t_load_start = time.perf_counter()
    tts = InstrumentedHebrewTTS(config, providers)
    t_load = time.perf_counter() - t_load_start
    print(f"[INIT] Model load time: {t_load*1000:.0f} ms")

    # ── Warm-up ──────────────────────────────────────────────────────────────
    # Always warm up (unless --no_warmup) so TensorRT has time to compile
    # engines and CUDA kernels are cached before timed runs.
    if getattr(args, 'warmup', True):
        print("\n[WARMUP] Running warm-up pass (TRT engine build may take a while)…")
        try:
            _wup_tts = tts  # use same instance so TRT engines are JIT-compiled
            _ = _wup_tts.infer(
                args.text,
                style_json_path=voice_abs,
                steps=min(args.steps, 4),
                cfg_scale=args.cfg,
                speed=args.speed,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[WARN] Warm-up failed: {e}")
        tts.reset_timings()
        if isinstance(tts.phonikud, TimedPhonikud):
            tts.phonikud.reset()
        print("[WARMUP] Done — timed runs begin now.")

    # ── Benchmark runs ────────────────────────────────────────────────────────
    all_totals: List[float] = []
    final_wav  = None

    for run_i in range(1, args.runs + 1):
        print(f"\n{'='*90}")
        print(f"  INFERENCE RUN {run_i}/{args.runs}")
        print(f"{'='*90}")

        tts.reset_timings()
        if isinstance(tts.phonikud, TimedPhonikud):
            tts.phonikud.reset()

        wav, phonikud_time, session_times, t_total = run_pipeline(
            tts,
            args.text,
            voice_abs,
            args.steps,
            args.cfg,
            args.speed,
            args.seed,
        )

        print_report(
            text          = args.text,
            providers     = providers,
            wav           = wav,
            sample_rate   = tts.sample_rate,
            phonikud_time = phonikud_time,
            session_times = session_times,
            t_total       = t_total,
            steps         = args.steps,
            run_idx       = run_i,
            runs          = args.runs,
            tts           = tts,
        )

        all_totals.append(t_total)
        final_wav = wav

    # ── Summary across runs ──────────────────────────────────────────────────
    if args.runs > 1:
        sep = "─" * 90
        print(f"\n{sep}")
        print(f"  SUMMARY ACROSS {args.runs} RUNS")
        print(sep)
        print(f"  mean={np.mean(all_totals)*1000:.1f} ms  "
              f"min={np.min(all_totals)*1000:.1f} ms  "
              f"max={np.max(all_totals)*1000:.1f} ms  "
              f"std={np.std(all_totals)*1000:.1f} ms")
        print(sep)

    # ── Save output ───────────────────────────────────────────────────────────
    if not args.no_save and final_wav is not None and len(final_wav) > 0:
        try:
            import soundfile as sf
            out_path = os.path.join(SCRIPT_DIR, args.out)
            sf.write(out_path, final_wav, tts.sample_rate)
            print(f"\n[SAVED] {out_path}")
        except ImportError:
            # Fallback: write raw PCM as WAV using wave module
            import wave, struct
            out_path = os.path.join(SCRIPT_DIR, args.out)
            pcm = (np.clip(final_wav, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(tts.sample_rate)
                wf.writeframes(pcm.tobytes())
            print(f"\n[SAVED] {out_path}  (soundfile not installed — wrote raw 16-bit PCM WAV)")


# ---------------------------------------------------------------------------
# TensorRT vs CUDA comparison
# ---------------------------------------------------------------------------

def _build_tts_for_provider(provider_name: str, fp16: bool, config_kwargs: dict) -> InstrumentedHebrewTTS:
    """Helper: build an InstrumentedHebrewTTS with a specific provider set."""
    available = ort.get_available_providers()
    if provider_name == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            return None, []
        providers = [
            ("TensorrtExecutionProvider", _trt_provider_options(fp16)),
            ("CUDAExecutionProvider",     _cuda_provider_options()),
            "CPUExecutionProvider",
        ]
    else:  # cuda
        if "CUDAExecutionProvider" not in available:
            return None, []
        providers = [
            ("CUDAExecutionProvider", _cuda_provider_options()),
            "CPUExecutionProvider",
        ]

    config = TTSConfig(
        use_gpu=True,
        **config_kwargs,
    )
    tts = InstrumentedHebrewTTS(config, providers)
    return tts, providers


def _run_comparison(args, onnx_dir_abs, phonikud_abs, tts_cfg_abs, voice_abs):
    """Run the pipeline under TensorRT and CUDA, then print a side-by-side table."""
    available = ort.get_available_providers()
    has_trt  = "TensorrtExecutionProvider" in available
    has_cuda = "CUDAExecutionProvider"     in available

    if not has_cuda and not has_trt:
        print("[ERROR] Neither TensorRT nor CUDA is available — comparison not possible.")
        return

    config_kwargs = dict(
        onnx_dir      = onnx_dir_abs,
        config_path   = tts_cfg_abs,
        phonikud_path = phonikud_abs,
        use_int8      = False,
        steps         = args.steps,
        cfg_scale     = args.cfg,
        speed         = args.speed,
        seed          = args.seed,
    )

    results = {}  # provider_label -> {totals, session_times, phonikud_time, audio_dur}

    for label, pname in [("TensorRT", "tensorrt"), ("CUDA", "cuda")]:
        print(f"\n{'='*90}")
        print(f"  LOADING — {label}")
        print(f"{'='*90}")
        tts, providers = _build_tts_for_provider(pname, args.fp16, config_kwargs)
        if tts is None:
            print(f"[SKIP] {label} not available.")
            continue

        pnames = [p if isinstance(p, str) else p[0] for p in providers]
        print(f"[INFO] {label} providers: {pnames}")

        # Warmup
        if args.warmup:
            print(f"[WARMUP] {label} — building TRT engines / caching CUDA kernels…")
            try:
                _ = tts.infer(
                    args.text, style_json_path=voice_abs,
                    steps=min(args.steps, 4), cfg_scale=args.cfg,
                    speed=args.speed, seed=args.seed,
                )
            except Exception as e:
                print(f"[WARN] Warm-up ({label}) failed: {e}")
            tts.reset_timings()
            if isinstance(tts.phonikud, TimedPhonikud):
                tts.phonikud.reset()
            print(f"[WARMUP] {label} ready.")

        # Timed runs
        totals: List[float] = []
        all_ph: List[float] = []
        all_sess: Dict[str, List[float]] = {}
        audio_dur = 0.0

        for run_i in range(1, args.runs + 1):
            tts.reset_timings()
            if isinstance(tts.phonikud, TimedPhonikud):
                tts.phonikud.reset()

            wav, phonikud_time, session_times, t_total = run_pipeline(
                tts, args.text, voice_abs,
                args.steps, args.cfg, args.speed, args.seed,
            )
            totals.append(t_total)
            all_ph.append(phonikud_time)
            for k, v in session_times.items():
                all_sess.setdefault(k, []).append(v)
            audio_dur = len(wav) / tts.sample_rate if len(wav) > 0 else 0.0

            print_report(
                text=args.text, providers=providers, wav=wav,
                sample_rate=tts.sample_rate, phonikud_time=phonikud_time,
                session_times=session_times, t_total=t_total,
                steps=args.steps, run_idx=run_i, runs=args.runs, tts=tts,
            )

        results[label] = {
            "totals":       totals,
            "ph_times":     all_ph,
            "sess_times":   all_sess,
            "audio_dur":    audio_dur,
            "sample_rate":  tts.sample_rate,
        }

        # Save last WAV
        if not args.no_save and len(wav) > 0:
            try:
                import soundfile as sf
                out = os.path.join(SCRIPT_DIR, f"{pname}_{args.out}")
                sf.write(out, wav, tts.sample_rate)
                print(f"[SAVED] {out}")
            except Exception:
                pass

    # ── Comparison table ─────────────────────────────────────────────────────
    if len(results) < 2:
        print("[INFO] Only one provider available — no comparison possible.")
        return

    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  TENSORRT vs CUDA — COMPARISON  (steps={args.steps}, runs={args.runs})")
    print(sep)

    labels = list(results.keys())
    print(f"  {'Stage':<35}  {labels[0]:>14}  {labels[1]:>14}  {'Speedup':>10}")
    print(f"  {'-'*35}  {'-'*14}  {'-'*14}  {'-'*10}")

    # Phonikud
    ph = [np.mean(results[l]["ph_times"]) * 1000 for l in labels]
    spd = ph[1] / ph[0] if ph[0] > 0 else float("inf")
    print(f"  {'phonikud':<35}  {ph[0]:>12.1f}ms  {ph[1]:>12.1f}ms  {spd:>9.2f}x")

    # Per session
    all_keys = set()
    for l in labels:
        all_keys.update(results[l]["sess_times"].keys())
    for key in all_keys:
        vals = [np.mean(results[l]["sess_times"].get(key, [0])) * 1000 for l in labels]
        spd  = vals[1] / vals[0] if vals[0] > 0 else float("inf")
        print(f"  {key:<35}  {vals[0]:>12.1f}ms  {vals[1]:>12.1f}ms  {spd:>9.2f}x")

    # Total
    tots = [np.mean(results[l]["totals"]) * 1000 for l in labels]
    rtfs = [tots[i] / 1000 / results[labels[i]]["audio_dur"] if results[labels[i]]["audio_dur"] > 0 else float("inf") for i in range(2)]
    spd  = tots[1] / tots[0] if tots[0] > 0 else float("inf")
    print(f"  {'-'*35}  {'-'*14}  {'-'*14}  {'-'*10}")
    print(f"  {'TOTAL (mean)':<35}  {tots[0]:>12.1f}ms  {tots[1]:>12.1f}ms  {spd:>9.2f}x")
    print(f"  {'RTF (lower=faster)':<35}  {rtfs[0]:>14.4f}  {rtfs[1]:>14.4f}")
    print(sep)
    winner = labels[0] if tots[0] < tots[1] else labels[1]
    print(f"  Winner: {winner}  ({min(tots):.1f} ms vs {max(tots):.1f} ms)")
    print(sep)


if __name__ == "__main__":
    main()
