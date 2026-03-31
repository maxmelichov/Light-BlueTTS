
import os
import sys
import random
import json

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from data.text_vocab import CHAR_TO_ID
from models.utils import LinearMelSpectrogram, compress_latents
from models.autoencoder.latent_encoder import LatentEncoder
from models.text2latent.dp_network import DPNetwork


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _load_dp_config(config_path: str) -> dict:
    """
    Load `configs/tts.json` and return a DP training config dict.

    We prefer the top-level "dp" section (new format). If it's missing, we
    fall back to a minimal compatibility config derived from "ttl".
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        full = json.load(f)

    dp_cfg = full.get("dp")
    if isinstance(dp_cfg, dict) and dp_cfg:
        return {"_full": full, **dp_cfg}

    # Back-compat fallback: use TTL compression defaults for DP.
    ttl_cfg = full.get("ttl", {}) if isinstance(full.get("ttl"), dict) else {}
    return {
        "_full": full,
        "latent_dim": ttl_cfg.get("latent_dim", 24),
        "chunk_compress_factor": ttl_cfg.get("chunk_compress_factor", 6),
        "normalizer": {"scale": ttl_cfg.get("normalizer", {}).get("scale", 1.0)},
        # Defaults matching `TTSDurationModel` paper implementation.
        "style_encoder": {"style_token_layer": {"n_style": 8, "style_value_dim": 16}},
    }


def collate_with_repeat_same_file(
    batch,
    sr: int = 44100,
    repeat_p: float = 0.3,
    sep_id: int = None,
    n_min: int = 2,
    n_max: int = 10,
    silence_sec: float = 0.2,
    spk2idx: dict = None,          # NEW: raw_spk_id -> contiguous idx
    unknown_spk: int = 0,          # NEW: fallback index if raw id not found
    max_total_samples: int = None, # cap total wav length after repetition
):
    """
    Repeat the SAME sample N times to synthesize longer sequences.

    batch items: (wav_1d[T], text_ids[L], speaker_id:int)
    returns: wavs_padded, texts_padded, text_masks, wav_lengths, speaker_ids_idx
    """
    import random
    import torch

    assert sep_id is not None and sep_id != 0
    assert 2 <= n_min <= n_max
    assert spk2idx is not None, "Pass spk2idx via functools.partial(...)"

    B = len(batch)
    num_rep = int(B * repeat_p)
    num_normal = B - num_rep

    # Unpack once
    wavs = [b[0].reshape(-1) for b in batch]  # ensure 1D
    texts = [b[1] for b in batch]
    speaker_ids_raw = [b[2] for b in batch]

    # Map raw speaker IDs to contiguous indices (stable)
    speaker_ids = []
    for s in speaker_ids_raw:
        if s in spk2idx:
             speaker_ids.append(int(spk2idx[s]))
        else:
             # Try int cast if key is missing (e.g. string "1" vs int 1)
             try:
                 s_int = int(s)
                 if s_int in spk2idx:
                     speaker_ids.append(int(spk2idx[s_int]))
                 else:
                     speaker_ids.append(unknown_spk)
             except (ValueError, TypeError):
                 speaker_ids.append(unknown_spk)

    # Shuffle indices so normal/repeat is unbiased
    idxs = list(range(B))
    random.shuffle(idxs)

    # Pre-create separator token ONCE (CPU)
    t_dtype = texts[0].dtype
    sep_tok = torch.tensor([sep_id], dtype=t_dtype)

    silence_len = int(silence_sec * sr)
    silence = torch.zeros(silence_len, dtype=wavs[0].dtype) if silence_len > 0 else None

    new_wavs = []
    new_texts = []
    new_speaker_ids = []

    # 1) Normal samples
    for i in idxs[:num_normal]:
        new_wavs.append(wavs[i])
        new_texts.append(texts[i])
        new_speaker_ids.append(speaker_ids[i])

    # 2) Repeated samples (repeat SAME sample => same speaker)
    for _ in range(num_rep):
        idx0 = idxs[random.randrange(B)]
        w0 = wavs[idx0]
        t0 = texts[idx0]
        spk = speaker_ids[idx0]

        N = random.randint(n_min, n_max)

        # Cap N so the total waveform stays under max_total_samples
        if max_total_samples is not None and w0.numel() > 0:
            max_N = max(1, max_total_samples // w0.numel())
            N = min(N, max_N)

        # ---- wav repeat ----
        if silence is None:
            w_cat = w0.repeat(N)
        else:
            total_len = N * w0.numel() + (N - 1) * silence_len
            w_cat = torch.empty(total_len, dtype=w0.dtype)
            pos = 0
            for k in range(N):
                w_cat[pos:pos + w0.numel()] = w0
                pos += w0.numel()
                if k < N - 1:
                    w_cat[pos:pos + silence_len] = silence
                    pos += silence_len

        # ---- text repeat ----
        if N == 1:
            t_cat = t0
        else:
            L = t0.numel()
            total_L = N * L + (N - 1)
            t_cat = torch.empty(total_L, dtype=t0.dtype)
            pos = 0
            sep_val = int(sep_tok.item())
            for k in range(N):
                t_cat[pos:pos + L] = t0
                pos += L
                if k < N - 1:
                    t_cat[pos] = sep_val
                    pos += 1

        new_wavs.append(w_cat)
        new_texts.append(t_cat)
        new_speaker_ids.append(spk)

    # ---- padding ----
    max_wav_len = max(w.numel() for w in new_wavs)
    max_text_len = max(t.numel() for t in new_texts)
    out_B = len(new_wavs)

    wavs_padded = torch.zeros(out_B, 1, max_wav_len, dtype=new_wavs[0].dtype)
    wav_lengths = torch.empty(out_B, dtype=torch.long)

    texts_padded = torch.zeros(out_B, max_text_len, dtype=new_texts[0].dtype)  # PAD=0
    text_masks = torch.zeros(out_B, 1, max_text_len, dtype=torch.float32)

    for i, (w, t) in enumerate(zip(new_wavs, new_texts)):
        wl = w.numel()
        tl = t.numel()
        wavs_padded[i, 0, :wl] = w
        wav_lengths[i] = wl
        texts_padded[i, :tl] = t
        text_masks[i, 0, :tl] = 1.0

    speaker_ids_tensor = torch.tensor(new_speaker_ids, dtype=torch.long)
    return wavs_padded, texts_padded, text_masks, wav_lengths, speaker_ids_tensor


def collate_dp(batch, spk2idx=None, unknown_spk=0):
    """
    Simple collate for duration predictor training (paper Sec 4.2).

    No sample repetition — each utterance is treated individually.
    The DP predicts utterance-level total latent duration.

    batch items from Text2LatentDataset:
        (wav, text_ids, speaker_id, ref_wav, is_self_ref, ref_speaker_id)
    returns:
        wavs_padded [B,1,T], texts_padded [B,L], text_masks [B,1,L],
        wav_lengths [B], speaker_ids [B]
    """
    wavs = [b[0].reshape(-1) for b in batch]
    texts = [b[1] for b in batch]
    speaker_ids_raw = [b[2] for b in batch]

    # Map raw speaker IDs to contiguous indices
    speaker_ids = []
    for s in speaker_ids_raw:
        if spk2idx is not None and s in spk2idx:
            speaker_ids.append(int(spk2idx[s]))
        elif spk2idx is not None:
            try:
                s_int = int(s)
                speaker_ids.append(int(spk2idx.get(s_int, unknown_spk)))
            except (ValueError, TypeError):
                speaker_ids.append(unknown_spk)
        else:
            speaker_ids.append(int(s))

    B = len(wavs)
    max_wav_len = max(w.numel() for w in wavs)
    max_text_len = max(t.numel() for t in texts)

    wavs_padded = torch.zeros(B, 1, max_wav_len, dtype=wavs[0].dtype)
    wav_lengths = torch.empty(B, dtype=torch.long)
    texts_padded = torch.zeros(B, max_text_len, dtype=texts[0].dtype)
    text_masks = torch.zeros(B, 1, max_text_len, dtype=torch.float32)

    for i, (w, t) in enumerate(zip(wavs, texts)):
        wl = w.numel()
        tl = t.numel()
        wavs_padded[i, 0, :wl] = w
        wav_lengths[i] = wl
        texts_padded[i, :tl] = t
        text_masks[i, 0, :tl] = 1.0

    speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long)
    return wavs_padded, texts_padded, text_masks, wav_lengths, speaker_ids_tensor


def train_duration_predictor(
    checkpoint_dir: str = "checkpoints/duration_predictor",
    ae_checkpoint: str = "checkpoints/ae/ae_latest_newer.pt",
    stats_path: str = "stats_real_data.pt",
    config_path: str = "configs/tts.json",
    max_steps: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Initializing Duration Predictor training on {device}...")

    # =========================================================
    # Load DP Config from tts.json
    # =========================================================
    dp_cfg = _load_dp_config(config_path)
    full_cfg = dp_cfg.get("_full", {})

    ae_cfg_json = full_cfg.get("ae", {})
    ae_enc_json = ae_cfg_json.get("encoder", {})
    ae_spec_json = ae_enc_json.get("spec_processor", {})
    ae_hop_length = ae_spec_json.get("hop_length", 512)
    ae_sample_rate = ae_cfg_json.get("sample_rate", 44100)
    ae_n_fft = ae_spec_json.get("n_fft", 2048)
    ae_n_mels = ae_spec_json.get("n_mels", 228)

    latent_dim = int(dp_cfg.get("latent_dim", 24))
    chunk_compress_factor = int(dp_cfg.get("chunk_compress_factor", 6))
    normalizer_scale = float(dp_cfg.get("normalizer", {}).get("scale", 1.0))

    stl = dp_cfg.get("style_encoder", {}).get("style_token_layer", {})
    style_tokens = int(stl.get("n_style", 8))
    style_dim = int(stl.get("style_value_dim", 16))

    compressed_channels = latent_dim * chunk_compress_factor

    print(f"\n{'='*60}")
    print(f"DP Config loaded from: {config_path}")
    print(f"  Version: {full_cfg.get('tts_version', 'unknown')}")
    print(f"  Split: {full_cfg.get('split', 'unknown')}")
    print(f"  latent_dim={latent_dim}, chunk_compress_factor={chunk_compress_factor}")
    print(f"  compressed_channels={compressed_channels}")
    print(f"  normalizer_scale={normalizer_scale}")
    print(f"  style_tokens={style_tokens}, style_dim={style_dim}")
    print(f"{'='*60}\n")

    # 1) AE encoder (frozen)
    ae_cfg = {
        "ksz":              ae_enc_json.get("ksz", 7),
        "hdim":             ae_enc_json.get("hdim", 512),
        "intermediate_dim": ae_enc_json.get("intermediate_dim", 2048),
        "dilation_lst":     ae_enc_json.get("dilation_lst", [1] * 10),
        "odim":             ae_enc_json.get("odim", 24),
        "idim":             ae_enc_json.get("idim", 1253),
    }
    mel_spec = LinearMelSpectrogram(
        sample_rate=ae_sample_rate,
        n_fft=ae_n_fft,
        hop_length=ae_hop_length,
        n_mels=ae_n_mels,
    ).to(device)
    ae_encoder = LatentEncoder(cfg=ae_cfg).to(device)

    if os.path.exists(ae_checkpoint):
        print(f"Loading AE checkpoint from {ae_checkpoint}")
        ckpt = torch.load(ae_checkpoint, map_location="cpu")
        if "encoder" in ckpt:
            ae_encoder.load_state_dict(ckpt["encoder"])
        elif "state_dict" in ckpt:
            ae_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            ae_encoder.load_state_dict(ckpt)
    else:
        print("Warning: AE checkpoint not found. Latent quality may be poor.")

    ae_encoder.eval()
    ae_encoder.requires_grad_(False)
    mel_spec.eval()

    # 2) Duration Predictor (paper-style DPNetwork)
    # Keep vocab_size = 37 (Hebrew G2B + Punctuation)
    model = DPNetwork(vocab_size=37, style_tokens=style_tokens, style_dim=style_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 3) Dataset
    metadata_path = "generated_audio/combined_dataset_cleaned_real_data.csv"
    dataset = Text2LatentDataset(
        metadata_path,
        sample_rate=ae_sample_rate,
        max_wav_len=ae_sample_rate * 20,
        max_text_len=800,
    )

    # Calculate weights for sampling
    speaker_ids = dataset.speaker_ids
    unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
    freq = dict(zip(unique_speakers, counts))
    print(f"Speaker counts: {freq}")
    
    # NEW: Build spk2idx mapping
    try:
        spk_raw = np.array(speaker_ids, dtype=np.int64)
        uniq = np.unique(spk_raw)
        spk2idx = {int(s): int(i) for i, s in enumerate(uniq)}
    except Exception as e:
        print(f"Warning: Could not cast speaker_ids to int64 ({e}). Using raw values.")
        uniq = np.unique(speaker_ids)
        spk2idx = {s: int(i) for i, s in enumerate(uniq)}
        
    num_speakers = len(uniq)
    print("num_speakers mapped:", num_speakers)
    
    weights = np.array([1.0 / freq[s] for s in speaker_ids], dtype=np.float32)
    weights = torch.from_numpy(weights)
    
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    
    # Paper Sec 4.2: DP is trained on individual utterances (no repetition).
    collate_fn = partial(collate_dp, spk2idx=spk2idx, unknown_spk=0)
    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
        import random
        random.seed(42 + worker_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 4) Latent stats
    if not os.path.exists(stats_path):
        print(f"Error: Stats file {stats_path} not found.")
        return

    stats = torch.load(stats_path, map_location=device)
    # Support both new stats (already [1,C,1]) and legacy flattened stats.
    if "mean" in stats and hasattr(stats["mean"], "dim") and stats["mean"].dim() == 3:
        mean = stats["mean"].to(device)
        std = stats["std"].to(device)
    else:
        mean = stats["mean"].to(device).view(1, -1, 1)
        std = stats["std"].to(device).view(1, -1, 1)

    if mean.shape[1] != compressed_channels:
        print(
            f"Warning: stats channels ({mean.shape[1]}) != expected compressed_channels "
            f"({compressed_channels} = {latent_dim}*{chunk_compress_factor})."
        )

    global_step = 0
    mean_loss = 0.0
    print("Starting DP training loop...")

    while global_step < max_steps:
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Step {global_step}/{max_steps}")

        for batch in progress_bar:
            if global_step >= max_steps:
                break

            wavs, text_ids, text_masks, lengths, speaker_ids = batch
            wavs = wavs.to(device)          # [B, 1, T_wav]
            text_ids = text_ids.to(device)  # [B, T_text]
            text_masks = text_masks.to(device)  # [B, 1, T_text]
            speaker_ids = speaker_ids.to(device) # [B]

            B = wavs.shape[0]

            # -------------------------------------------------
            # 1) Latent extraction + normalization
            # -------------------------------------------------
            with torch.no_grad():
                mel = mel_spec(wavs.squeeze(1))       # [B, 228, T_mel]
                z = ae_encoder(mel)                   # [B, 24, T_lat]
                z = compress_latents(z, factor=chunk_compress_factor)  # [B, Cc, T_lat_c]
                # normalized compressed latents (+ optional extra normalizer scale)
                z = ((z - mean) / std) * normalizer_scale

            B, C, T_lat = z.shape

            # Compute valid latent length from waveform length
            valid_mel_len = lengths.to(device).float() / ae_hop_length
            valid_z_len = (valid_mel_len / chunk_compress_factor).ceil().long().clamp(min=1, max=T_lat)

            # -------------------------------------------------
            # 2) Build reference segments (5% to 95% of speech)
            # -------------------------------------------------
            # Move to CPU/numpy once to avoid per-sample GPU sync
            vz_np = valid_z_len.cpu().numpy()

            ref_list = []
            ref_len_list = []

            for i in range(B):
                L_i = int(vz_np[i])

                # Range [5%, 95%] of usable frames
                start_min = int(L_i * 0.05)
                start_max = int(L_i * 0.95)

                if start_max <= start_min:
                    start = 0
                    end = L_i
                else:
                    start = random.randint(start_min, start_max - 1)
                    max_end = max(start + 1, int(L_i * 0.95))
                    seg_len = random.randint(1, max_end - start)
                    end = start + seg_len

                ref_list.append((i, start, end))
                ref_len_list.append(end - start)

            max_ref_len = max(ref_len_list)
            z_ref = torch.zeros(B, C, max_ref_len, device=device)
            ref_mask = torch.zeros(B, 1, max_ref_len, device=device)

            for i, s, e in ref_list:
                L_ref = e - s
                z_ref[i, :, :L_ref] = z[i, :, s:e]
                ref_mask[i, :, :L_ref] = 1.0

            # -------------------------------------------------
            # 3) Forward DPNetwork (in LOG domain)
            # -------------------------------------------------
            log_pred = model(
                text_ids=text_ids,
                z_ref=z_ref,
                text_mask=text_masks,
                ref_mask=ref_mask,
                return_log=True,
            )  # [B]

            if global_step == 0:
                print(f"[Sanity] Pred shape: {log_pred.shape}") # Should be [B]
                print("raw speaker examples:", dataset.speaker_ids[:10])
                print("mapped speaker examples:", speaker_ids[:10])
                print("unique mapped:", speaker_ids.unique())

            # -------------------------------------------------
            # 4) Target and loss (in LOG domain)
            # -------------------------------------------------
            dur_gt = valid_z_len.float()  # [B]
            log_gt = torch.log(dur_gt.clamp(min=1e-5))

            # L1 loss on log duration
            loss = F.l1_loss(log_pred, log_gt)

            if global_step % 20 == 0:
                # Sanity check print (convert back to linear for display)
                pred_linear = torch.exp(log_pred[:4]).detach()
                gt_linear = dur_gt[:4].detach()
                print(f"\n[Step {global_step}]")
                print(f"  Pred (Lin): {pred_linear.cpu().numpy()}")
                print(f"  Target:     {gt_linear.cpu().numpy()}")
                print(f"  Loss (Log): {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            mean_loss += loss.item()
            progress_bar.set_postfix(loss=mean_loss / (global_step + 1), global_step=global_step)

            if global_step % 500 == 0:
                save_path = os.path.join(
                    checkpoint_dir, f"duration_predictor_{global_step}.pt"
                )
                torch.save(model.state_dict(), save_path)
                print(f"Saved DP checkpoint to {save_path}")

    # Final save
    final_path = os.path.join(checkpoint_dir, "duration_predictor_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Duration Predictor training complete. Saved to {final_path}")


if __name__ == "__main__":
    set_seed(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tts.json",
        help="Path to tts.json config file (default: configs/tts.json)",
    )
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_duration_predictor(
        config_path=args.config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=(args.device if args.device is not None else ("cuda:1" if torch.cuda.is_available() else "cpu")),
    )