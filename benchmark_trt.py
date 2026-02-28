import os
import sys
import tensorrt as trt
import torch
import numpy as np
import time
import argparse
import soundfile as sf
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_vocab import text_to_indices
from utils import load_ttl_config

# Create a logger
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, namespace="")

class TRTEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def run(self, inputs):
        """
        inputs: dict of name -> torch.Tensor
        returns: dict of name -> torch.Tensor (outputs)
        """
        bindings = [0] * self.engine.num_io_tensors
        
        # Set input shapes and bindings
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise ValueError(f"Missing input: {name}")
                tensor = inputs[name]
                # Ensure contiguous
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                
                # Set shape for dynamic inputs
                self.context.set_input_shape(name, tensor.shape)
                bindings[i] = tensor.data_ptr()
            else:
                # Output binding - we need to allocate memory based on output shape
                pass
        
        # Validate shapes are set
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all binding shapes specified")

        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                
                # Map TRT dtype to Torch dtype
                if dtype == trt.float32:
                    torch_dtype = torch.float32
                elif dtype == trt.float16:
                    torch_dtype = torch.float16
                elif dtype == trt.int32:
                    torch_dtype = torch.int32
                elif dtype == trt.int64:
                    torch_dtype = torch.int64
                elif dtype == trt.bool:
                    torch_dtype = torch.bool
                elif dtype == trt.int8:
                    torch_dtype = torch.int8
                else:
                    raise TypeError(f"Unsupported dtype: {dtype}")
                
                # Allocate output tensor
                out_tensor = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
                bindings[i] = out_tensor.data_ptr()
                outputs[name] = out_tensor

        # Execute
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        
        return outputs

    def input_names(self):
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def output_names(self):
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

def load_style_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    style_ttl = np.array(data["style_ttl"]["data"], dtype=np.float32)
    style_keys = np.array(data["style_keys"]["data"], dtype=np.float32)
    style_dp = None
    if "style_dp" in data:
        style_dp = np.array(data["style_dp"]["data"], dtype=np.float32)
    
    return torch.from_numpy(style_ttl), torch.from_numpy(style_keys), (torch.from_numpy(style_dp) if style_dp is not None else None)

def load_uncond(onnx_dir="onnx_models"):
    """Load unconditional embeddings for CFG (same as ONNX inference)."""
    uncond_path = os.path.join(onnx_dir, "uncond.npz")
    if os.path.exists(uncond_path):
        uncond = np.load(uncond_path)
        u_text = torch.from_numpy(uncond['u_text'].astype(np.float32))
        u_ref = torch.from_numpy(uncond['u_ref'].astype(np.float32))
        u_keys = torch.from_numpy(uncond['u_keys'].astype(np.float32)) if 'u_keys' in uncond.files else None
        return u_text, u_ref, u_keys
    else:
        print("[WARN] uncond.npz not found. CFG will use zeros (not recommended).")
        return None, None, None

def benchmark(text_input, z_ref_path, out_wav, style_json_path=None, steps=32, cfg_scale=3.0, speed=1.0, cfg=None):
    print("Loading Engines...")
    try:
        ref_enc = None
        if os.path.exists("trt_engines/reference_encoder.trt"):
             ref_enc = TRTEngine("trt_engines/reference_encoder.trt")
        elif style_json_path is None:
             print("Warning: Reference Encoder TRT not found and no style_json provided.")
             
        text_enc = TRTEngine("trt_engines/text_encoder.trt")
        
        dp = None
        if os.path.exists("trt_engines/lenght_pred.trt"):
            dp = TRTEngine("trt_engines/lenght_pred.trt")

        dp_style = None
        if os.path.exists("trt_engines/lenght_predictor_style.trt"):
            dp_style = TRTEngine("trt_engines/lenght_predictor_style.trt")

        vf = TRTEngine("trt_engines/backbone.trt")
        vocoder = TRTEngine("trt_engines/vocoder.trt")
    except Exception as e:
        print(f"Failed to load engines: {e}")
        return

    device = "cuda"
    B = 1

    # Config-derived dimensions
    compressed_channels = cfg["compressed_channels"] if cfg else 144
    latent_dim = cfg["latent_dim"] if cfg else 24
    chunk_compress_factor = cfg["chunk_compress_factor"] if cfg else 6
    normalizer_scale = cfg["normalizer_scale"] if cfg else 1.0
    
    # 0. Prep Input
    ids = text_to_indices(text_input)
    text_ids = torch.tensor([ids], dtype=torch.int64, device=device)
    T_text = text_ids.shape[1]
    text_mask = torch.ones(B, 1, T_text, dtype=torch.float32, device=device)
    
    # Load unconditional embeddings for CFG
    u_text, u_ref, u_keys = load_uncond()
    if u_text is not None:
        u_text = u_text.to(device)
        u_ref = u_ref.to(device)
        u_keys = u_keys.to(device) if u_keys is not None else None
    
    # Prep Reference
    ref_values = None
    ref_keys = None
    style_dp = None
    t_ref = 0
    z_ref_is_normalized = False
    z_ref = None

    # Load z_ref (needed for DP in all cases unless style_dp is used)
    if z_ref_path and os.path.exists(z_ref_path):
        print(f"Loading z_ref from {z_ref_path}")
        payload = torch.load(z_ref_path, map_location=device)
        if isinstance(payload, dict):
            # Prefer z_ref_raw when available. Older .pt files often contain z_ref_norm
            # without normalizer_scale applied, which is out-of-distribution and can
            # cause DP to explode and audio to become noise.
            if "z_ref_raw" in payload:
                z_ref = payload["z_ref_raw"].to(device)
                z_ref_is_normalized = False
            elif "z_ref_norm" in payload:
                z_ref = payload["z_ref_norm"].to(device)
                z_ref_is_normalized = True
                print("  [WARN] Using pre-normalized z_ref (z_ref_norm). This may be incompatible with normalizer_scale.")
            elif "z" in payload:
                z_ref = payload["z"].to(device)
            else:
                # Fallback to first tensor in dict
                z_ref = list(payload.values())[0].to(device)
        elif torch.is_tensor(payload):
            z_ref = payload.to(device)
            
        if z_ref.dim() == 2: 
            z_ref = z_ref.unsqueeze(0)

        # If raw, normalize with stats + normalizer_scale (match training/inference_tts.py)
        if z_ref is not None and not z_ref_is_normalized:
            if os.path.exists("stats_real_data.pt"):
                stats = torch.load("stats_real_data.pt", map_location=device)
                mean = stats["mean"].to(device)
                std = stats["std"].to(device)
                if mean.ndim == 1: mean = mean.view(1, -1, 1)
                if std.ndim == 1: std = std.view(1, -1, 1)
                z_ref = ((z_ref - mean) / std) * normalizer_scale
                z_ref_is_normalized = True
                print(f"  [INFO] Normalized z_ref_raw using stats_real_data.pt (normalizer_scale={normalizer_scale})")
            else:
                print("[WARN] stats_real_data.pt not found; using raw z_ref (quality may suffer).")

        # Match PyTorch inference preprocessing: trim tail ~5% and cap to 150 frames.
        if z_ref is not None and z_ref.ndim == 3:
            T = int(z_ref.shape[2])
            tail_trim = max(2, int(T * 0.05))
            T_trimmed = max(1, T - tail_trim)
            if T_trimmed < T:
                z_ref = z_ref[:, :, :T_trimmed]
            target_frames = 150
            if z_ref.shape[2] > target_frames:
                z_ref = z_ref[:, :, :target_frames]
    elif not style_json_path:
        print(f"Warning: z_ref path {z_ref_path} not found and no style_json. Using random.")
        z_ref = torch.randn(B, compressed_channels, 256, device=device)

    # Safety truncate z_ref for DP if needed (after preprocessing)
    if z_ref is not None and z_ref.shape[2] > 1024:
        z_ref = z_ref[:, :, :1024]
    
    if z_ref is not None:
        ref_mask = torch.ones(B, 1, z_ref.shape[2], dtype=torch.float32, device=device)
    else:
        ref_mask = None

    if style_json_path:
        print(f"Loading style from {style_json_path}")
        rv, rk, rdp = load_style_json(style_json_path)
        ref_values = rv.to(device)
        ref_keys = rk.to(device)
        if rdp is not None:
            style_dp = rdp.to(device)
            if style_dp.dim() == 2: style_dp = style_dp.unsqueeze(0)
            
        # Assuming batch size 1 for now in benchmark
        if ref_values.dim() == 2: ref_values = ref_values.unsqueeze(0)
        if ref_keys.dim() == 2: ref_keys = ref_keys.unsqueeze(0)

    elif ref_enc and z_ref is not None:
        # Ref Encoder requires fixed length 256 usually in our TRT exports
        TARGET_REF_LEN = 256
        z_ref_for_enc = z_ref.clone()
        if z_ref_for_enc.shape[2] < TARGET_REF_LEN:
            z_ref_for_enc = torch.nn.functional.pad(z_ref_for_enc, (0, TARGET_REF_LEN - z_ref_for_enc.shape[2]))
        else:
            z_ref_for_enc = z_ref_for_enc[:, :, :TARGET_REF_LEN]
        ref_mask_enc = torch.ones(B, 1, TARGET_REF_LEN, dtype=torch.float32, device=device)

        print(f"Benchmarking with T_text={T_text}, text: {text_input[:50]}...")

        # 1. Reference Encoder
        # ONNX signature: (z_ref, mask) -> (ref_values, ref_keys)
        ref_enc_inputs = {"z_ref": z_ref_for_enc, "mask": ref_mask_enc}
        # Warmup
        for _ in range(5):
            ref_out = ref_enc.run(ref_enc_inputs)
        
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            ref_out = ref_enc.run(ref_enc_inputs)
        torch.cuda.synchronize()
        t_ref = (time.time() - t0) / 50
        print(f"Reference Encoder: {t_ref*1000:.2f} ms")
        
        ref_values = ref_out["ref_values"]
        ref_keys = ref_out["ref_keys"]
    else:
        raise ValueError("No Reference Encoder TRT found and no style_json provided. Cannot proceed.")

    # 2. Text Encoder (TRT/ONNX signatures vary by export)
    # Common variants:
    # - (text_ids, style_ttl, text_mask) -> text_emb
    # - (text_ids, text_mask, ref_values, ref_keys) -> text_emb
    te_in = set(text_enc.input_names())
    text_enc_inputs = {"text_ids": text_ids}
    if "text_mask" in te_in:
        text_enc_inputs["text_mask"] = text_mask
    if "style_ttl" in te_in:
        text_enc_inputs["style_ttl"] = ref_values
    if "ref_values" in te_in:
        text_enc_inputs["ref_values"] = ref_values
    if "ref_keys" in te_in:
        text_enc_inputs["ref_keys"] = ref_keys
    # Warmup
    for _ in range(5):
        text_out = text_enc.run(text_enc_inputs)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        text_out = text_enc.run(text_enc_inputs)
    torch.cuda.synchronize()
    t_text = (time.time() - t0) / 50
    print(f"Text Encoder: {t_text*1000:.2f} ms")

    text_emb = text_out["text_emb"]

    # 3. Duration Predictor
    t_dp = 0
    dur_pred = None
    
    if dp_style and style_dp is not None:
        # Warmup
        for _ in range(5):
            dp_out = dp_style.run({
                "text_ids": text_ids,
                "style_dp": style_dp,
                "text_mask": text_mask
            })
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            dp_out = dp_style.run({
                "text_ids": text_ids,
                "style_dp": style_dp,
                "text_mask": text_mask
            })
        torch.cuda.synchronize()
        t_dp = (time.time() - t0) / 50
        print(f"Duration Predictor (Style): {t_dp*1000:.2f} ms")
        dur_pred = dp_out["duration"]
        
    elif dp and z_ref is not None:
        # Warmup
        for _ in range(5):
            dp_out = dp.run({
                "text_ids": text_ids,
                "z_ref": z_ref,
                "text_mask": text_mask,
                "ref_mask": ref_mask
            })
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            dp_out = dp.run({
                "text_ids": text_ids,
                "z_ref": z_ref,
                "text_mask": text_mask,
                "ref_mask": ref_mask
            })
        torch.cuda.synchronize()
        t_dp = (time.time() - t0) / 50
        print(f"Duration Predictor: {t_dp*1000:.2f} ms")
        dur_pred = dp_out["duration"]
    else:
        print("[WARN] No DP model available or missing inputs (z_ref/style_dp). Using fallback.")
    
    # Use DP output for T_lat (matching ONNX inference logic)
    if dur_pred is not None:
        raw_val = float(dur_pred.sum())
        
        # Handle Infinity/NaN
        if not np.isfinite(raw_val):
            print(f"[WARN] DP output infinite/NaN: {raw_val}. Using fallback.")
            T_lat = int(T_text * 1.3)
        else:
            # Match ONNX: T_lat = round(dur_pred / speed), then cap by token count
            T_lat = int(np.round(raw_val / max(speed, 1e-6)))
            T_lat = max(T_lat, 1)
            
            # Cap: IPA phonemes average ~0.85 frames/char; 3x is generous.
            txt_len = T_text
            T_cap = max(20, min(txt_len * 3 + 20, 600))
            T_lat = min(T_lat, T_cap)
            
            # Global cap (matches training code)
            T_lat = min(T_lat, 800)
            print(f"  [DEBUG] DP Raw: {raw_val:.2f} -> T_lat={T_lat} (speed={speed}, cap={T_cap})")
    else:
        T_lat = int(T_text * 1.3)
    
    T_lat = max(10, T_lat)
    
    # Vocoder TRT supports max [1, latent_dim, 2048], and vocoder_input = T_lat * ccf
    MAX_T_LAT = 2048 // chunk_compress_factor  # e.g. 2048/6 = 341
    if T_lat > MAX_T_LAT:
        print(f"[WARN] T_lat {T_lat} exceeds vocoder max ({MAX_T_LAT}). Clamping.")
        T_lat = MAX_T_LAT
    
    print(f"Predicted T_lat: {T_lat}")

    # 4. Vector Field Estimator (Loop N steps)
    #
    # TRT exports differ:
    # - Some engines output `denoised_latent` (next state).
    # - Others output `velocity` (vector field), in which case we must do:
    #     x_{k+1} = x_k + (1/total_step) * velocity
    noisy_latent = torch.randn(B, compressed_channels, T_lat, dtype=torch.float32, device=device)
    latent_mask = torch.ones(B, 1, T_lat, dtype=torch.float32, device=device)
    
    total_step = torch.tensor([float(steps)], dtype=torch.float32, device=device)
    
    # Setup unconditional inputs for CFG
    if u_text is not None and u_ref is not None:
        u_text_ctx = u_text  # [1, 256, 1]
        u_text_mask = torch.ones((1, 1, 1), dtype=torch.float32, device=device)
        u_style = u_ref  # [1, 50, 256]
    else:
        u_text_ctx = torch.zeros((1, text_emb.shape[1], 1), dtype=torch.float32, device=device)
        u_text_mask = torch.ones((1, 1, 1), dtype=torch.float32, device=device)
        u_style = torch.zeros_like(ref_values)

    vf_in = set(vf.input_names())
    vf_out_names = vf.output_names()
    has_denoised = "denoised_latent" in vf_out_names
    has_velocity = "velocity" in vf_out_names
    if not (has_denoised or has_velocity):
        raise ValueError(f"Unsupported backbone outputs: {vf_out_names}")

    style_mask = torch.ones(B, 1, ref_values.shape[1], dtype=torch.float32, device=device)

    def vf_inputs(noisy, txt_ctx, style_vals, style_k, txt_mask, step_val):
        """Build VF input dict matching the engine signature."""
        feed = {"noisy_latent": noisy}
        if "text_emb" in vf_in:
            feed["text_emb"] = txt_ctx
        if "text_context" in vf_in:
            feed["text_context"] = txt_ctx
        if "style_ttl" in vf_in:
            feed["style_ttl"] = style_vals
        if "style_keys" in vf_in:
            feed["style_keys"] = style_k
        if "latent_mask" in vf_in:
            feed["latent_mask"] = latent_mask
        if "text_mask" in vf_in:
            feed["text_mask"] = txt_mask
        if "style_mask" in vf_in:
            feed["style_mask"] = style_mask
        if "current_step" in vf_in:
            feed["current_step"] = torch.tensor([float(step_val)], dtype=torch.float32, device=device)
        if "total_step" in vf_in:
            feed["total_step"] = total_step
        return feed
    
    # Warmup
    for _ in range(5):
        vf_out = vf.run(vf_inputs(noisy_latent, text_emb, ref_values, ref_keys, text_mask, 0))

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10): 
        x = noisy_latent.clone()
        for s in range(steps):
            out = vf.run(vf_inputs(x, text_emb, ref_values, ref_keys, text_mask, s))
            if has_denoised:
                x = out["denoised_latent"]
            else:
                # velocity is diff_out; integrate with 1/total_step (matches vf_estimator.py ONNX path)
                x = x + (out["velocity"] / total_step.view(1, 1, 1))
    torch.cuda.synchronize()
    t_vf_total = (time.time() - t0) / 10
    print(f"Vector Estimator ({steps} steps): {t_vf_total*1000:.2f} ms")

    # Real Inference Pass for Saving Audio (with CFG)
    print(f"Performing one real sampling pass for audio saving (CFG={cfg_scale})...")
    latent = torch.randn(B, compressed_channels, T_lat, dtype=torch.float32, device=device)
    
    for s in range(steps):
        cond_out_dict = vf.run(vf_inputs(latent, text_emb, ref_values, ref_keys, text_mask, s))
        if has_denoised:
            cond_next = cond_out_dict["denoised_latent"]
        else:
            cond_next = latent + (cond_out_dict["velocity"] / total_step.view(1, 1, 1))
        
        # CFG: Unconditional pass
        if cfg_scale != 1.0:
            u_k = u_keys if u_keys is not None else ref_keys
            uncond_out_dict = vf.run(vf_inputs(latent, u_text_ctx, u_style, u_k, u_text_mask, s))
            if has_denoised:
                uncond_next = uncond_out_dict["denoised_latent"]
            else:
                uncond_next = latent + (uncond_out_dict["velocity"] / total_step.view(1, 1, 1))
            # CFG on next-state outputs: x_next = uncond + scale * (cond - uncond)
            latent = uncond_next + cfg_scale * (cond_next - uncond_next)
        else:
            latent = cond_next


    # Load stats for denormalization (check multiple locations like ONNX inference)
    mean = None
    std = None
    stats_paths = ["onnx_models/stats.npz", "trt_engines/stats.npz", "stats_real_data.pt"]
    for sp in stats_paths:
        if os.path.exists(sp):
            if sp.endswith(".npz"):
                stats = np.load(sp)
                mean = torch.from_numpy(stats['mean'].astype(np.float32)).view(1, -1, 1).to(device)
                std = torch.from_numpy(stats['std'].astype(np.float32)).view(1, -1, 1).to(device)
            else:
                stats = torch.load(sp, map_location=device)
                mean = stats["mean"].view(1, -1, 1).to(device)
                std = stats["std"].view(1, -1, 1).to(device)
            print(f"Loaded stats from {sp}")
            break
    
    if mean is None:
        print("[WARNING] No stats found! Audio will be noise.")
        mean = torch.zeros(1, compressed_channels, 1, device=device)
        std = torch.ones(1, compressed_channels, 1, device=device)

    # 5. Vocoder
    # Un-normalize: reverse ((x - mean) / std) * normalizer_scale
    latent = ((latent / normalizer_scale) * std) + mean
    
    # Decompress latent: [1, C_lat, T] -> [1, latent_dim, T * chunk_compress_factor]
    # Matches decompress_latents() in models/utils.py: view -> permute -> flatten
    z_dec_in = latent.view(B, latent_dim, chunk_compress_factor, T_lat).permute(0, 1, 3, 2).reshape(B, latent_dim, T_lat * chunk_compress_factor)
    # Apply latent_mask to valid region before decompressing (zeros padded frames)
    
    # Warmup
    for _ in range(5):
        voc_out = vocoder.run({"latent": z_dec_in})
        
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        voc_out = vocoder.run({"latent": z_dec_in})
    torch.cuda.synchronize()
    t_voc = (time.time() - t0) / 50
    print(f"Vocoder: {t_voc*1000:.2f} ms")

    # Save Audio
    wav = voc_out["waveform"].squeeze().cpu().numpy()
    ae_sr = cfg["ae_sample_rate"] if cfg else 44100
    sf.write(out_wav, wav, ae_sr)
    print(f"Saved audio to {out_wav}")

    # Total Latency and RTF
    total_latency = t_ref + t_text + t_dp + t_vf_total + t_voc
    num_samples = wav.shape[-1]
    sr = ae_sr
    audio_dur = num_samples / sr
    rtf = total_latency / audio_dur
    
    print("-" * 30)
    print(f"Total Latency: {total_latency*1000:.2f} ms")
    print(f"Generated Audio: {audio_dur:.2f} s ({num_samples} samples)")
    print(f"RTF: {rtf:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="jˈeʃ ʁeɡaʔˈim ʃebahˈem haʔolˈam ʔotsˈeʁ liʃnijˈa, lˈo kˈi kaʁˈa mˈaʃehu dʁamˈati, ʔˈela kˈi hevˈanu pitʔˈom mˈaʃehu χadˈaʃ. kˈaχ bedijˈuk niʁʔˈejt hatkufˈa ʃebˈa ʔˈanu χajˈim. ʁaʔjonˈot ʃenoldˈu kaχalˈom hofχˈim limetsiʔˈut ʃemeʔatsˈevet meχadˈaʃ ʔˈet hadˈeʁeχ ʃebˈa ʔˈanu medabʁˈim, jotsʁˈim umkablˈim haχlatˈot.")
    parser.add_argument("--z_ref", type=str, default=None)
    parser.add_argument("--style_json", type=str, default=None, help="Path to style JSON to skip ReferenceEncoder")
    parser.add_argument("--out", type=str, default="benchmark_out.wav")
    parser.add_argument("--steps", type=int, default=32, help="Denoising steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier Free Guidance scale")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (higher = faster/shorter)")
    parser.add_argument("--config", type=str, default="tts.json", help="Path to tts.json config")
    args = parser.parse_args()

    # Load config
    ttl_cfg = None
    if os.path.exists(args.config):
        ttl_cfg = load_ttl_config(args.config)
        print(f"[INFO] Loaded config: {args.config} (v{ttl_cfg['full_config'].get('tts_version', '?')})")
    else:
        print(f"[WARN] Config {args.config} not found, using hardcoded defaults.")
    
    if args.z_ref is None and args.style_json is None:
        # Try default
        if os.path.exists("male1.pt"):
            args.z_ref = "male1.pt"
        else:
            print("Error: Must provide --z_ref or --style_json")
            sys.exit(1)

    benchmark(args.text, args.z_ref, args.out, style_json_path=args.style_json, 
              steps=args.steps, cfg_scale=args.cfg, speed=args.speed, cfg=ttl_cfg)
