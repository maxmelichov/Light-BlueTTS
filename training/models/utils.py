import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

def compress_latents(z: torch.Tensor, factor: int = 6) -> torch.Tensor:
    """
    Compress latent sequence by grouping 'factor' frames.
    Input: [B, C, T]
    Output: [B, C * factor, T // factor]
    """
    B, C, T = z.shape
    # Pad if necessary
    if T % factor != 0:
        pad = factor - (T % factor)
        z = torch.nn.functional.pad(z, (0, pad))
        T = T + pad
        
    z = z.view(B, C, T // factor, factor)
    z = z.permute(0, 1, 3, 2)             # [B, 24, 6, T_low]
    z = z.flatten(1, 2)                   # [B, 144, T_low]
    return z

def decompress_latents(z: torch.Tensor, factor: int = 6, target_channels: int = 24) -> torch.Tensor:
    """
    Decompress latents (inverse of compress_latents).
    Input: [B, 144, T_low]
    Output: [B, 24, T_high]
    """
    B, C_total, T_low = z.shape
    # z: [B, 144, T_low]
    # Unflatten 144 -> 24, 6
    z = z.view(B, target_channels, factor, T_low) # [B, 24, 6, T_low]
    # Permute back to [B, 24, T_low, 6]
    z = z.permute(0, 1, 3, 2)
    # Reshape to [B, 24, T_high]
    z = z.flatten(2, 3) # [B, 24, 6*T_low]
    return z

def _resolve_vocab_size(char_dict_path, default=37):
    """Try loading a char_dict JSON to determine vocab_size.  Falls back to *default*."""
    import json as _json
    import os as _os
    if char_dict_path and _os.path.exists(char_dict_path):
        try:
            with open(char_dict_path, "r") as f:
                cd = _json.load(f)
            # char_dict is typically {char: id, ...}
            vs = max(cd.values()) + 1 if isinstance(cd, dict) else len(cd)
            return vs
        except Exception:
            pass
    return default


def load_ttl_config(config_path="configs/tts.json"):
    """
    Load tts.json and return a flat dict of derived parameters for *all*
    modules (TTL, AE, DP).

    Used by training, inference, export, and benchmark scripts so that model
    construction and dimension constants come from one config file.
    """
    import json
    with open(config_path, "r") as f:
        full_config = json.load(f)
    ttl = full_config["ttl"]
    ae = full_config.get("ae", {})
    dp = full_config.get("dp", {})
    te = ttl["text_encoder"]
    se = ttl["style_encoder"]
    vf = ttl["vector_field"]
    um = ttl["uncond_masker"]

    # Vocab size – try char_dict, fall back to 37 (original IPA set)
    char_dict_path = te.get("char_dict_path", te.get("text_embedder", {}).get("char_dict_path"))
    vocab_size = _resolve_vocab_size(char_dict_path, default=37)

    # DP vocab size (may differ from TTL text encoder)
    dp_char_dict_path = dp.get("sentence_encoder", {}).get("char_dict_path",
                               dp.get("sentence_encoder", {}).get("text_embedder", {}).get("char_dict_path"))
    dp_vocab_size = _resolve_vocab_size(dp_char_dict_path, default=vocab_size)

    # AE decoder config dict (for LatentDecoder1D construction)
    ae_dec = ae.get("decoder", {})
    ae_dec_cfg = {
        "idim": ae_dec.get("idim", 24),
        "hdim": ae_dec.get("hdim", 512),
        "intermediate_dim": ae_dec.get("intermediate_dim", 2048),
        "ksz": ae_dec.get("ksz", 7),
        "dilation_lst": ae_dec.get("dilation_lst", [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]),
        "chunk_compress_factor": ae.get("chunk_compress_factor", 1),
        "head": {
            "idim": ae_dec.get("head", {}).get("idim", ae_dec.get("hdim", 512)),
            "hdim": ae_dec.get("head", {}).get("hdim", 2048),
            "odim": ae_dec.get("head", {}).get("odim", 512),
            "ksz": ae_dec.get("head", {}).get("ksz", 3),
        },
    }

    # AE encoder config dict (for LatentEncoder construction)
    ae_enc = ae.get("encoder", {})
    ae_enc_spec = ae_enc.get("spec_processor", {})
    ae_enc_cfg = {
        "ksz": ae_enc.get("ksz", 7),
        "hdim": ae_enc.get("hdim", 512),
        "intermediate_dim": ae_enc.get("intermediate_dim", 2048),
        "dilation_lst": ae_enc.get("dilation_lst", [1] * 10),
        "odim": ae_enc.get("odim", 24),
        "idim": ae_enc.get("idim", 1253),
    }

    # DP style encoder params
    dp_se = dp.get("style_encoder", {}).get("style_token_layer", {})

    return {
        "full_config": full_config,
        "ttl": ttl,
        "ae": ae,
        "dp": dp,
        # ---- Vocab ----
        "vocab_size": vocab_size,
        "char_dict_path": char_dict_path,
        "dp_vocab_size": dp_vocab_size,
        # ---- Core dims (Paper Sec 3.2.1) ----
        "latent_dim": ttl["latent_dim"],
        "chunk_compress_factor": ttl["chunk_compress_factor"],
        "compressed_channels": ttl["latent_dim"] * ttl["chunk_compress_factor"],
        # ---- Normalizer ----
        "normalizer_scale": ttl["normalizer"]["scale"],
        # ---- Flow matching ----
        "sigma_min": ttl["flow_matching"]["sig_min"],
        # ---- Batch expander ----
        "Ke": ttl["batch_expander"]["n_batch_expand"],
        # ---- Text encoder ----
        "te_d_model": te["text_embedder"]["char_emb_dim"],
        "te_convnext_layers": te["convnext"]["num_layers"],
        "te_expansion_factor": te["convnext"]["intermediate_dim"] // te["text_embedder"]["char_emb_dim"],
        "te_attn_n_layers": te["attn_encoder"]["n_layers"],
        "te_attn_p_dropout": te["attn_encoder"]["p_dropout"],
        # ---- Style / Reference encoder ----
        "se_d_model": se["proj_in"]["odim"],
        "se_hidden_dim": se["convnext"]["intermediate_dim"],
        "se_num_blocks": se["convnext"]["num_layers"],
        "se_n_style": se["style_token_layer"]["n_style"],
        "se_n_heads": se["style_token_layer"]["n_heads"],
        # ---- Uncond masker / CFG ----
        "prob_both_uncond": um["prob_both_uncond"],
        "prob_text_uncond": um["prob_text_uncond"],
        "uncond_init_std": um["std"],
        "um_text_dim": um["text_dim"],
        "um_n_style": um["n_style"],
        "um_style_key_dim": um["style_key_dim"],
        "um_style_value_dim": um["style_value_dim"],
        # ---- Vector field ----
        "vf_hidden": vf["proj_in"]["odim"],
        "vf_time_dim": vf["time_encoder"]["time_dim"],
        "vf_n_blocks": vf["main_blocks"]["n_blocks"],
        "vf_text_dim": vf["main_blocks"]["text_cond_layer"]["text_dim"],
        "vf_text_n_heads": vf["main_blocks"]["text_cond_layer"]["n_heads"],
        "vf_style_dim": vf["main_blocks"]["style_cond_layer"]["style_dim"],
        "vf_rotary_scale": vf["main_blocks"]["text_cond_layer"]["rotary_scale"],
        # ---- AE decoder config (for LatentDecoder1D) ----
        "ae_dec_cfg": ae_dec_cfg,
        # ---- AE encoder config (for LatentEncoder / export_ref_latent) ----
        "ae_enc_cfg": ae_enc_cfg,
        "ae_sample_rate": ae.get("sample_rate", 44100),
        "ae_n_fft": ae_enc_spec.get("n_fft", 2048),
        "ae_hop_length": ae_enc_spec.get("hop_length", 512),
        "ae_n_mels": ae_enc_spec.get("n_mels", 1253),
        # ---- DP params ----
        "dp_style_tokens": dp_se.get("n_style", 8),
        "dp_style_dim": dp_se.get("style_value_dim", 16),
    }


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=1253,
        f_min=0,
        f_max=None,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=1.0,
        )
    
    def forward(self, audio):
        mel = self.mel(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if mel.dim() == 4 and mel.shape[1] == 1:
             mel = mel.squeeze(1)
        return mel

class MelSpectrogramNoLog(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=1253,
        f_min=0,
        f_max=12000,
        power=1.0,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=power,
        )

    def forward(self, audio):
        mel = self.mel(audio)
        # No log here
        if mel.dim() == 4 and mel.shape[1] == 1:
            mel = mel.squeeze(1)
        return mel

class LinearMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=1253,
        f_min=0,
        f_max=None,
    ):
        super().__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            power=1.0,
        )
        self.mel_scale = T.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
        )

    def forward(self, audio):
        spec = self.spectrogram(audio)
        mel = self.mel_scale(spec)
        
        spec = torch.log(torch.clamp(spec, min=1e-5))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        # Concatenate along the channel/frequency dimension (dim=1 for [B, C, T])
        return torch.cat([spec, mel], dim=1)