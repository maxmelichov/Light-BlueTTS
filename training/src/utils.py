import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from models.utils import decompress_latents


def build_reference_only(z_ref_input, valid_z_ref_len, device):
    _, _, T_ref = z_ref_input.shape
    arange_T = torch.arange(T_ref, device=device).unsqueeze(0)
    valid_len = valid_z_ref_len.clamp(min=0, max=T_ref).unsqueeze(1)
    ref_mask_left = (arange_T < valid_len).unsqueeze(1).float()
    z_ref_left = z_ref_input * ref_mask_left
    return z_ref_left, ref_mask_left


def build_reference_from_latents(z_1, valid_z_len, z_ref_input, valid_z_ref_len, is_self_ref, device, chunk_compress_factor=6, hop_length=512, sample_rate=44100):
    # arxiv.org/pdf/2512.17293 Sec 4.2
    B, C, T = z_1.shape
    _, _, T_ref_in = z_ref_input.shape
    compressed_rate = (sample_rate / hop_length) / chunk_compress_factor
    min_frames = max(1, int(round(0.2 * compressed_rate)))
    max_frames = int(round(9.0 * compressed_rate))

    z_ref_left = torch.zeros(B, C, T, device=device)
    ref_mask_left = torch.zeros(B, 1, T, device=device)
    target_loss_mask = torch.ones(B, 1, T, device=device)
    train_T_lat = valid_z_len.clone()

    for i in range(B):
        sample_T = int(valid_z_len[i].item())
        ref_T = int(valid_z_ref_len[i].item())
        ref_T = min(ref_T, T_ref_in)

        if is_self_ref[i]:
            sample_T = int(valid_z_len[i].item())
            half_len = max(1, sample_T // 2)
            upper_bound = max(1, min(max_frames, half_len))
            if upper_bound < min_frames:
                 length = int(torch.randint(1, upper_bound + 1, (1,), device=device).item())
            else:
                 length = int(torch.randint(min_frames, upper_bound + 1, (1,), device=device).item())
            length = min(length, sample_T)
            if length < 1: length = 1

            max_start = max(0, sample_T - length)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            mask_start = start
            mask_end = min(start + length, sample_T)
            target_loss_mask[i, :, mask_start:mask_end] = 0.0
            copy_len = min(length, T)
            z_ref_left[i, :, :copy_len] = z_1[i, :, mask_start:mask_start + copy_len]
            ref_mask_left[i, :, :copy_len] = 1.0

        else:
            half_ref = max(1, ref_T // 2)
            upper_bound = max(1, min(max_frames, half_ref))
            if upper_bound < min_frames:
                 length = int(torch.randint(1, upper_bound + 1, (1,), device=device).item())
            else:
                 length = int(torch.randint(min_frames, upper_bound + 1, (1,), device=device).item())
            length = min(length, ref_T)
            if length < 1: length = 1
            max_start = max(0, ref_T - length)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            copy_len = min(length, T)
            z_ref_left[i, :, :copy_len] = z_ref_input[i, :, start:start+copy_len]
            ref_mask_left[i, :, :copy_len] = 1.0

    return z_ref_left, ref_mask_left, train_T_lat, target_loss_mask


@torch.no_grad()
def sample_audio(
    vf_estimator,
    text_encoder,
    reference_encoder,
    ae_decoder,
    text_ids,
    text_mask,
    z_ref,
    ref_enc_mask,
    u_text,
    u_ref,
    u_keys,
    mean,
    std,
    duration_predictor=None,
    steps=32,
    cfg_scale=1.75,
    device='cuda',
    debug_label=None,
    speed=1.0,
    style_ttl=None,
    style_keys=None,
    style_dp=None,
    latent_dim=24,
    chunk_compress_factor=6,
    normalizer_scale=1.0,
    hop_length=512,
):
    if debug_label:
        print(f"[{debug_label}] Starting sampling...")

    B = text_ids.shape[0]
    C = latent_dim * chunk_compress_factor

    if style_ttl is not None:
        ref_values = style_ttl
        _ = style_keys if style_keys is not None else style_ttl
    else:
        ref_values, _ = reference_encoder(z_ref, mask=ref_enc_mask)

    if duration_predictor is not None:
        dur_pred = duration_predictor(
            text_ids,
            z_ref=z_ref,
            text_mask=text_mask,
            ref_mask=ref_enc_mask,
            style_tokens=style_dp,
            return_log=True,
        )

        T_lat = (torch.exp(dur_pred) / speed).clamp(min=1).round().long()

        if text_mask.ndim == 3 and text_mask.shape[1] == 1:
            txt_len = text_mask.sum(dim=(1,2)).long()
        else:
             txt_len = text_mask.sum().long() // text_mask.shape[0]
             if text_mask.ndim == 2:
                 txt_len = text_mask.sum(dim=1).long()
        T_cap = (txt_len * 3 + 20).clamp(min=20, max=600)
        T_lat = torch.minimum(T_lat, T_cap)

        if debug_label:
             print(f"[{debug_label}] DP T_lat: {T_lat.cpu().numpy()}")

        T_lat = T_lat.clamp(max=800)
        T = max(int(T_lat.max().item()), 10)
        latent_mask = (
            torch.arange(T, device=device)
            .expand(B, T) < T_lat.unsqueeze(1)
        ).unsqueeze(1).float()
    else:
        T = z_ref.shape[2] if z_ref is not None else 200
        latent_mask = torch.ones(B, 1, T, device=device)

    h_text = text_encoder(text_ids, ref_values, text_mask=text_mask)

    if isinstance(text_encoder, DDP):
         vf_style_keys = text_encoder.module.ref_keys
    else:
         vf_style_keys = text_encoder.ref_keys
    vf_style_keys = vf_style_keys.expand(B, -1, -1)

    h_text_null = u_text.expand(B, -1, 1)
    h_ref_null = u_ref.expand(B, -1, -1)
    h_keys_null = u_keys.expand(B, -1, -1) if u_keys is not None else vf_style_keys

    x = torch.randn(B, C, T, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)
        x_in = x * latent_mask
        v_cond = vf_estimator(
            noisy_latent=x_in,
            text_emb=h_text,
            style_ttl=ref_values,
            style_keys=vf_style_keys,
            latent_mask=latent_mask,
            text_mask=text_mask,
            current_step=t,
        )

        if cfg_scale != 1.0:
            v_uncond = vf_estimator(
                noisy_latent=x_in,
                text_emb=h_text_null,
                style_ttl=h_ref_null,
                style_keys=h_keys_null,
                latent_mask=latent_mask,
                text_mask=torch.ones(B, 1, 1, device=device),
                current_step=t,
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        v = v * latent_mask
        x = (x + v * dt) * latent_mask

    if normalizer_scale != 1.0 and normalizer_scale != 0.0:
        z_pred = (x / normalizer_scale) * std + mean
    else:
        z_pred = x * std + mean

    z_pred = decompress_latents(z_pred, factor=chunk_compress_factor, target_channels=latent_dim)
    wav_pred = ae_decoder(z_pred)
    frame_len = hop_length * chunk_compress_factor
    wav_pred = wav_pred[..., frame_len:-frame_len]
    return wav_pred


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
