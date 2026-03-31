import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from models.autoencoder.latent_encoder import LatentEncoder
from models.utils import LinearMelSpectrogram, compress_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tts-json",
        default=os.path.join(os.path.dirname(__file__), "configs", "tts.json"),
        help="Path to the TTS config json (default: configs/tts.json)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 0. Load TTS config (single source of truth)
    # ------------------------------------------------------------------
    tts_json_path = args.tts_json
    if not os.path.exists(tts_json_path):
        print(f"ERROR: TTS config not found: {tts_json_path}")
        return

    with open(tts_json_path, "r", encoding="utf-8") as f:
        tts_cfg = json.load(f)

    if "ae" not in tts_cfg or "encoder" not in tts_cfg["ae"] or "spec_processor" not in tts_cfg["ae"]["encoder"]:
        print(f"ERROR: Unexpected tts.json format (missing ae.encoder.spec_processor): {tts_json_path}")
        return

    spec_cfg = tts_cfg["ae"]["encoder"]["spec_processor"]
    ae_enc_cfg = tts_cfg["ae"]["encoder"]
    compression_factor = int(tts_cfg.get("ttl", {}).get("chunk_compress_factor", 6))
    sample_rate = int(tts_cfg.get("ae", {}).get("sample_rate", spec_cfg.get("sample_rate", 44100)))
    hop_length = int(spec_cfg.get("hop_length", 512))

    print(f"Using TTS config: {tts_json_path}")

    # ------------------------------------------------------------------
    # 1. Resolve metadata path
    # ------------------------------------------------------------------
    # Preferred: your Hebrew phoneme dataset, since that's what T2L uses.
    metadata_candidates = [
        "generated_audio/combined_dataset_cleaned_real_data.csv",
    ]

    metadata_path = None
    for p in metadata_candidates:
        if os.path.exists(p):
            metadata_path = p
            break

    if metadata_path is None:
        print("ERROR: No metadata CSV found in known locations.")
        print("Checked:")
        for p in metadata_candidates:
            print("  -", p)
        return

    print(f"Using metadata: {metadata_path}")

    # Paths
    checkpoint_path = "checkpoints/ae/ae_latest_newer.pt"
    cfg_ckpt = tts_cfg.get("ae_ckpt_path")
    if isinstance(cfg_ckpt, str) and cfg_ckpt and cfg_ckpt != "unknown.pt":
        checkpoint_path = cfg_ckpt
    output_path = "stats_real_data.pt"

    # ------------------------------------------------------------------
    # 2. Configs (must match your T2L training)
    # ------------------------------------------------------------------
    mel_args = {
        "sample_rate": int(spec_cfg.get("sample_rate", sample_rate)),
        "n_fft": int(spec_cfg.get("n_fft", 2048)),
        "hop_length": hop_length,
        "n_mels": int(spec_cfg.get("n_mels", 228)),
    }

    ae_cfg = {
        "ksz": int(ae_enc_cfg.get("ksz", 7)),
        "hdim": int(ae_enc_cfg.get("hdim", 512)),
        "intermediate_dim": int(ae_enc_cfg.get("intermediate_dim", 2048)),
        "dilation_lst": ae_enc_cfg.get("dilation_lst", [1] * int(ae_enc_cfg.get("num_layers", 10))),
        "odim": int(ae_enc_cfg.get("odim", 24)),   # uncompressed AE latent channels
        "idim": int(ae_enc_cfg.get("idim", 1253)),
    }

    latent_dim = int(ae_cfg["odim"] * compression_factor)
    print(
        "AE/mel settings from tts.json: "
        f"sr={sample_rate}, hop={hop_length}, n_mels={mel_args['n_mels']}, "
        f"odim={ae_cfg['odim']}, Kc={compression_factor}, latent_dim={latent_dim}"
    )

    # ------------------------------------------------------------------
    # 3. Dataset + DataLoader (use same class as T2L)
    # ------------------------------------------------------------------
    print("Loading dataset...")
    dataset = Text2LatentDataset(
        metadata_path=metadata_path,
        sample_rate=sample_rate,
        max_wav_len=sample_rate * 20,
        max_text_len=None,  # text not needed for stats
    )
    print(f"Dataset size for stats: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_text2latent,
    )

    # ------------------------------------------------------------------
    # 4. Load AE encoder + Mel frontend (exactly as in T2L)
    # ------------------------------------------------------------------
    mel_spec = LinearMelSpectrogram(**mel_args).to(device)
    encoder = LatentEncoder(cfg=ae_cfg).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading AE checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        elif "state_dict" in ckpt:
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            try:
                encoder.load_state_dict(ckpt)
            except Exception as e:
                print(f"Could not load AE encoder state dict properly: {e}")
    else:
        print(
            f"WARNING: Checkpoint {checkpoint_path} not found. "
            "Using random encoder weights – stats will be meaningless."
        )

    encoder.eval()
    mel_spec.eval()

    # ------------------------------------------------------------------
    # 5. Accumulate per-channel stats on COMPRESSED latents [B, latent_dim, Tz]
    # ------------------------------------------------------------------
    total_sum = torch.zeros(latent_dim, device=device)
    total_sq_sum = torch.zeros(latent_dim, device=device)
    total_frames = 0

    print("Computing latent stats over dataset...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Text2LatentDataset collate returns: 
            # wavs, texts, text_masks, lengths, speaker_ids, ref_wavs, ref_lengths, is_self_ref, ref_speaker_ids
            wavs = batch[0]
            lengths = batch[3]
            
            wavs = wavs.to(device)  # [B, 1, T_wav] (per your training script)

            # 1) Mel spectrogram
            # In T2L you use: mel = mel_spec(wavs.squeeze(1))
            mel = mel_spec(wavs.squeeze(1))  # [B, 228, T_mel]

            # 2) Encode latents
            z = encoder(mel)  # [B, 24, T_enc]

            # 3) Temporal compression to match T2L pipeline
            z = compress_latents(z, factor=compression_factor)  # [B, 144, T_z]

            B, C, T_z = z.shape

            # 4) Valid length masking (same logic as T2L training)
            # mel hop: hop_length, compression: Kc → valid_mel_len / Kc = valid_z_len
            valid_mel_len = lengths.to(device).float() / float(hop_length)
            valid_z_len = (valid_mel_len / compression_factor).ceil().long()
            valid_z_len = valid_z_len.clamp(min=1, max=T_z)

            # Build mask [B, T_z]
            time_idx = torch.arange(T_z, device=device).expand(B, T_z)
            mask = time_idx < valid_z_len.unsqueeze(1)  # [B, T_z] bool

            # Flatten masked latents
            # z: [B, C, T_z] → [B, T_z, C]
            z_bt = z.permute(0, 2, 1).contiguous()  # [B, T_z, latent_dim]
            valid_z = z_bt[mask]  # [N_valid, latent_dim]

            if valid_z.numel() == 0:
                continue

            total_sum += valid_z.sum(dim=0)
            total_sq_sum += (valid_z ** 2).sum(dim=0)
            total_frames += valid_z.shape[0]

    if total_frames == 0:
        print("ERROR: No valid frames found while computing stats.")
        return

    # ------------------------------------------------------------------
    # 6. Final mean/std
    # ------------------------------------------------------------------
    mean = total_sum / total_frames  # [144]
    var = (total_sq_sum / total_frames) - mean ** 2
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)            # [144]

    print(f"Global mean (avg over channels): {mean.mean().item():.4f}")
    print(f"Global std  (avg over channels): {std.mean().item():.4f}")

    stats = {
        "mean": mean.cpu(),       # [latent_dim]
        "std": std.cpu(),         # [latent_dim]
        "Kc": compression_factor,
        "latent_dim": latent_dim,
    }

    torch.save(stats, output_path)
    print(f"Saved latent stats to {output_path}")


if __name__ == "__main__":
    main()