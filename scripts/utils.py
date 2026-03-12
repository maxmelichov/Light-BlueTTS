"""Shared utilities for Light-BlueTTS inference and tooling."""

import json
import os
from typing import Any, Dict, Optional


def load_ttl_config(config_path: str) -> Dict[str, Any]:
    """Load tts.json and return a flat dict with commonly used model dimensions.

    The returned dict mirrors the keys expected by ``create_tensorrt.py``,
    ``benchmark_trt.py``, and ``obfuscate_onnx.py``.

    Args:
        config_path: Path to the ``tts.json`` configuration file.

    Returns:
        Dictionary with extracted model parameters:
            - full_config: the raw parsed JSON
            - latent_dim, chunk_compress_factor, compressed_channels
            - normalizer_scale
            - vocab_size (text encoder vocab)
            - te_d_model, te_convnext_layers, te_attn_n_layers,
              te_expansion_factor, te_attn_p_dropout
            - se_d_model, se_hidden_dim, se_num_blocks, se_n_style, se_n_heads
            - vf_hidden, vf_text_dim, vf_style_dim, vf_n_blocks,
              vf_time_dim, vf_rotary_scale
            - ae_sample_rate, ae_dec_cfg, hop_length
            - dp_vocab_size, dp_style_tokens, dp_style_dim
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ttl = cfg.get("ttl", {})
    ae = cfg.get("ae", {})
    dp_cfg = cfg.get("dp", {})

    latent_dim = int(ttl.get("latent_dim", 24))
    chunk_compress_factor = int(ttl.get("chunk_compress_factor", 6))
    compressed_channels = latent_dim * chunk_compress_factor

    normalizer_scale = float(ttl.get("normalizer", {}).get("scale", 1.0))

    # Text encoder
    te = ttl.get("text_encoder", {})
    te_convnext = te.get("convnext", {})
    te_attn = te.get("attn_encoder", {})
    te_d_model = int(te_convnext.get("idim", 256))

    # Style encoder
    se = ttl.get("style_encoder", {})
    se_convnext = se.get("convnext", {})
    stl = se.get("style_token_layer", {})
    se_d_model = int(se_convnext.get("idim", 256))
    se_n_style = int(stl.get("n_style", 50))

    # Vector field estimator
    vf = ttl.get("vector_field", {})
    vf_proj = vf.get("proj_in", {})
    vf_time = vf.get("time_encoder", {})
    vf_blocks = vf.get("main_blocks", {})
    vf_text_cond = vf_blocks.get("text_cond_layer", {})

    # AE / Vocoder
    ae_enc = ae.get("encoder", {})
    ae_spec = ae_enc.get("spec_processor", {})
    ae_sample_rate = int(ae.get("sample_rate", 44100))
    hop_length = int(ae_spec.get("hop_length", 512))

    # Duration Predictor
    dp_se = dp_cfg.get("style_encoder", {})
    dp_stl = dp_se.get("style_token_layer", {})

    # Vocab size: count chars in the char_dict if available, else use default
    vocab_size = 40  # safe default matching text_vocab.py

    return {
        "full_config": cfg,
        # Latent space
        "latent_dim": latent_dim,
        "chunk_compress_factor": chunk_compress_factor,
        "compressed_channels": compressed_channels,
        "normalizer_scale": normalizer_scale,
        # Text Encoder
        "vocab_size": vocab_size,
        "te_d_model": te_d_model,
        "te_convnext_layers": int(te_convnext.get("num_layers", 6)),
        "te_attn_n_layers": int(te_attn.get("n_layers", 4)),
        "te_expansion_factor": int(te_convnext.get("intermediate_dim", 1024)) // te_d_model,
        "te_attn_p_dropout": float(te_attn.get("p_dropout", 0.0)),
        # Style Encoder (Reference Encoder)
        "se_d_model": se_d_model,
        "se_hidden_dim": int(se_convnext.get("intermediate_dim", 1024)),
        "se_num_blocks": int(se_convnext.get("num_layers", 6)),
        "se_n_style": se_n_style,
        "se_n_heads": int(stl.get("n_heads", 2)),
        # Vector Field Estimator
        "vf_hidden": int(vf_proj.get("odim", 512)),
        "vf_text_dim": int(vf_text_cond.get("text_dim", 256)),
        "vf_style_dim": int(vf_blocks.get("style_cond_layer", {}).get("style_dim", 256)),
        "vf_n_blocks": int(vf_blocks.get("n_blocks", 4)),
        "vf_time_dim": int(vf_time.get("time_dim", 64)),
        "vf_rotary_scale": float(vf_text_cond.get("rotary_scale", 10)),
        # AutoEncoder / Vocoder
        "ae_sample_rate": ae_sample_rate,
        "ae_dec_cfg": ae.get("decoder", {}),
        "hop_length": hop_length,
        # Duration Predictor
        "dp_vocab_size": vocab_size,
        "dp_style_tokens": int(dp_stl.get("n_style", 8)),
        "dp_style_dim": int(dp_stl.get("style_value_dim", 64)),
    }
