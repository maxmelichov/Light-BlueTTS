import os
import sys
import glob
import json
import random
import numpy as np
import soundfile as sf
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from data.audio_utils import ensure_sr
from data.text_vocab import text_to_indices
from models.autoencoder.latent_encoder import LatentEncoder
from models.autoencoder.latent_decoder import LatentDecoder1D
from models.utils import LinearMelSpectrogram, compress_latents
from models.text2latent.text_encoder import TextEncoder
from models.text2latent.vf_estimator import VectorFieldEstimator
from models.text2latent.reference_encoder import ReferenceEncoder
from models.text2latent.dp_network import DPNetwork
from utils import build_reference_only, build_reference_from_latents, sample_audio, seed_worker

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class UncondParams(nn.Module):
    """Learnable unconditional tokens for CFG. Dims from ttl.uncond_masker config."""
    def __init__(self, text_dim=256, n_style=50, style_key_dim=256, style_value_dim=256, init_std=0.1):
        super().__init__()
        self.u_text = nn.Parameter(torch.randn(1, text_dim, 1) * init_std)
        self.u_ref = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)
        self.u_keys = nn.Parameter(torch.randn(1, n_style, style_key_dim) * init_std)

def train(
    checkpoint_dir="checkpoints/text2latent",
    ae_checkpoint="checkpoints/ae/ae_latest_newer.pt",
    stats_path="stats_real_data.pt",
    config_path="configs/tts.json",  # Path to tts.json config
    epochs=1000,
    batch_size=16,
    lr=5e-4,
    Ke=None,  # Context-sharing expansion factor (None = use config ttl.batch_expander.n_batch_expand)
    puncond=None, # CFG dropout probability (None = use config ttl.uncond_masker probs)
    device="cuda:1" if torch.cuda.is_available() else "cpu",
    finetune=False,  # Finetune mode: lr=1e-4, SPFM starts after 5K steps
    accumulation_steps=1
):
    # DDP Init
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if rank == 0:
            print(f"Initialized DDP on Rank {rank}")
    else:
        rank = 0
        local_rank = 0
        if isinstance(device, str):
            device = torch.device(device)
        print(f"Running on single device {device} (No DDP env found)")

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_dir = os.path.join(checkpoint_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        print(f"Initializing training on {device}...")
    else:
        log_dir = os.path.join(checkpoint_dir, "logs")

    # Finetune mode overrides
    if finetune:
        lr = 5e-4
        spfm_start_override = 40_000
        if rank == 0:
            print(f"[Finetune Mode] lr={lr}, SPFM warm-up={spfm_start_override} steps")
    else:
        spfm_start_override = None

    # =========================================================
    # Load TTL Config from tts.json
    # =========================================================
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    ttl_cfg = full_config["ttl"]
    ae_cfg_json = full_config.get("ae", {})
    ae_enc_json = ae_cfg_json.get("encoder", {})
    ae_spec_json = ae_enc_json.get("spec_processor", {})
    ae_hop_length = ae_spec_json.get("hop_length", 512)
    ae_sample_rate = ae_cfg_json.get("sample_rate", 44100)
    ae_n_fft = ae_spec_json.get("n_fft", 2048)
    ae_n_mels = ae_spec_json.get("n_mels", 228)
    
    latent_dim = ttl_cfg["latent_dim"]
    chunk_compress_factor = ttl_cfg["chunk_compress_factor"]
    compressed_channels = latent_dim * chunk_compress_factor

    cfg_Ke = ttl_cfg["batch_expander"]["n_batch_expand"]
    if Ke is None:
        Ke = cfg_Ke

    normalizer_scale = ttl_cfg["normalizer"]["scale"]
    sigma_min = ttl_cfg["flow_matching"]["sig_min"]

    te_cfg = ttl_cfg["text_encoder"]
    te_d_model = te_cfg["text_embedder"]["char_emb_dim"]
    te_convnext_layers = te_cfg["convnext"]["num_layers"]
    te_convnext_intermediate = te_cfg["convnext"]["intermediate_dim"]
    te_expansion_factor = te_convnext_intermediate // te_d_model
    te_attn_n_heads = te_cfg["attn_encoder"]["n_heads"]
    te_attn_n_layers = te_cfg["attn_encoder"]["n_layers"]
    te_attn_filter_channels = te_cfg["attn_encoder"]["filter_channels"]
    te_attn_p_dropout = te_cfg["attn_encoder"]["p_dropout"]

    se_cfg = ttl_cfg["style_encoder"]
    se_d_model = se_cfg["proj_in"]["odim"]
    se_hidden_dim = se_cfg["convnext"]["intermediate_dim"]
    se_num_blocks = se_cfg["convnext"]["num_layers"]
    se_n_style = se_cfg["style_token_layer"]["n_style"]
    se_n_heads = se_cfg["style_token_layer"]["n_heads"]

    spte_cfg = ttl_cfg["speech_prompted_text_encoder"]
    spte_n_heads = spte_cfg["n_heads"]
    spte_n_style = se_n_style

    um_cfg = ttl_cfg["uncond_masker"]
    prob_both_uncond = um_cfg["prob_both_uncond"]
    prob_text_uncond = um_cfg["prob_text_uncond"]
    uncond_init_std = um_cfg["std"]
    um_text_dim = um_cfg["text_dim"]
    um_n_style = um_cfg["n_style"]
    um_style_key_dim = um_cfg["style_key_dim"]
    um_style_value_dim = um_cfg["style_value_dim"]

    if puncond is None:
        puncond = prob_both_uncond + prob_text_uncond

    vf_cfg = ttl_cfg["vector_field"]
    vf_hidden = vf_cfg["proj_in"]["odim"]
    vf_time_dim = vf_cfg["time_encoder"]["time_dim"]
    vf_n_blocks = vf_cfg["main_blocks"]["n_blocks"]
    vf_text_dim = vf_cfg["main_blocks"]["text_cond_layer"]["text_dim"]
    vf_text_n_heads = vf_cfg["main_blocks"]["text_cond_layer"]["n_heads"]
    vf_style_dim = vf_cfg["main_blocks"]["style_cond_layer"]["style_dim"]
    vf_rotary_scale = vf_cfg["main_blocks"]["text_cond_layer"]["rotary_scale"]
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TTL Config loaded from: {config_path}")
        print(f"  Version: {full_config.get('tts_version', 'unknown')}")
        print(f"  Split: {full_config.get('split', 'unknown')}")
        print(f"  latent_dim={latent_dim}, chunk_compress_factor={chunk_compress_factor}")
        print(f"  compressed_channels={compressed_channels}")
        print(f"  Ke={Ke} (config: {cfg_Ke})")
        print(f"  normalizer_scale={normalizer_scale}")
        print(f"  sigma_min={sigma_min}")
        print(f"  TextEncoder: d_model={te_d_model}, conv_layers={te_convnext_layers}, "
              f"attn_layers={te_attn_n_layers}, p_dropout={te_attn_p_dropout}")
        print(f"  ReferenceEncoder: d_model={se_d_model}, blocks={se_num_blocks}, "
              f"n_style={se_n_style}, n_heads={se_n_heads}")
        print(f"  VF Estimator: hidden={vf_hidden}, blocks={vf_n_blocks}, "
              f"time_dim={vf_time_dim}, rotary_scale={vf_rotary_scale}")
        print(f"  Uncond: prob_both={prob_both_uncond}, prob_text={prob_text_uncond}, "
              f"init_std={uncond_init_std}, total_puncond={puncond}")
        print(f"{'='*60}\n")

    # 1. Load Stats
    if not os.path.exists(stats_path):
        print(f"Error: Stats file {stats_path} not found. Run compute_latent_stats.py first.")
        return
    
    stats = torch.load(stats_path, map_location=device)
    if "mean" in stats and stats["mean"].dim() == 3:
        mean = stats["mean"].to(device)
        std = stats["std"].to(device)
    else:
        # Fallback for old stats files
        mean = stats['mean'].to(device).view(1, -1, 1)
        std = stats['std'].to(device).view(1, -1, 1)
    
    ref_wav_path_v1 = "/home/maxm/AE_training_data_all/slow_44K/data/real_data/yoav_times/recording_id002/chunk_0002_7.4-19.6s.wav"
    if os.path.exists(ref_wav_path_v1):
        print(f"Loading inference reference for Voice 1 from {ref_wav_path_v1}")
        ref_wav_np, sr = sf.read(ref_wav_path_v1)
        ref_wav_torch_v1 = torch.from_numpy(ref_wav_np).float().to(device)
        if ref_wav_torch_v1.dim() > 1: ref_wav_torch_v1 = ref_wav_torch_v1.mean(dim=1)
        if sr != ae_sample_rate:
            ref_wav_torch_v1 = ensure_sr(ref_wav_torch_v1, sr, ae_sample_rate, device=device)
        else:
            ref_wav_torch_v1 = ref_wav_torch_v1.unsqueeze(0)
        if ref_wav_torch_v1.dim() == 2 and ref_wav_torch_v1.size(0) != 1:
            ref_wav_torch_v1 = ref_wav_torch_v1.mean(dim=0, keepdim=True)
        elif ref_wav_torch_v1.dim() == 1:
            ref_wav_torch_v1 = ref_wav_torch_v1.unsqueeze(0)
    else:
        print(f"Warning: Inference reference for Voice 1 {ref_wav_path_v1} not found.")
        ref_wav_torch_v1 = None
    
    ae_cfg = {
        "ksz":              ae_enc_json.get("ksz", 7),
        "hdim":             ae_enc_json.get("hdim", 512),
        "intermediate_dim": ae_enc_json.get("intermediate_dim", 2048),
        "dilation_lst":     ae_enc_json.get("dilation_lst", [1] * 10),
        "odim":             ae_enc_json.get("odim", 24),
        "idim":             ae_enc_json.get("idim", 1253),
    }
    mel_spec = LinearMelSpectrogram(sample_rate=ae_sample_rate, n_fft=ae_n_fft, hop_length=ae_hop_length, n_mels=ae_n_mels).to(device)
    ae_encoder = LatentEncoder(cfg=ae_cfg).to(device)

    ae_dec_json = ae_cfg_json.get("decoder", {})
    ae_dec_cfg = {
        "idim":             ae_dec_json.get("idim", 24),
        "hdim":             ae_dec_json.get("hdim", 512),
        "intermediate_dim": ae_dec_json.get("intermediate_dim", 2048),
        "ksz":              ae_dec_json.get("ksz", 7),
        "dilation_lst":     ae_dec_json.get("dilation_lst", [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]),
        "head":             ae_dec_json.get("head", {"hdim": 2048, "odim": 512}),
    }
    ae_decoder = LatentDecoder1D(cfg=ae_dec_cfg).to(device)

    if os.path.exists(ae_checkpoint):
        print(f"Loading AE checkpoint from {ae_checkpoint}")
        ckpt = torch.load(ae_checkpoint, map_location='cpu')
        if 'encoder' in ckpt:
            ae_encoder.load_state_dict(ckpt['encoder'])
        elif 'state_dict' in ckpt:
            ae_encoder.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            try: ae_encoder.load_state_dict(ckpt)
            except: print("Warning: Could not load AE Encoder weights cleanly.")

        if 'decoder' in ckpt:
            ae_decoder.load_state_dict(ckpt['decoder'])
        else:
            print("Warning: 'decoder' key not found in AE checkpoint.")
    else:
        print("Warning: AE Checkpoint not found!")

    ae_encoder.eval()
    ae_encoder.requires_grad_(False)
    ae_decoder.eval()
    ae_decoder.requires_grad_(False)
    mel_spec.eval()

    text_encoder = TextEncoder(
        vocab_size=37,
        d_model=te_d_model,
        n_conv_layers=te_convnext_layers,
        n_attn_layers=te_attn_n_layers,
        expansion_factor=te_expansion_factor,
        p_dropout=te_attn_p_dropout,
    ).to(device)

    reference_encoder = ReferenceEncoder(
        in_channels=compressed_channels,
        d_model=se_d_model,
        hidden_dim=se_hidden_dim,
        num_blocks=se_num_blocks,
        num_tokens=se_n_style,
        num_heads=se_n_heads,
    ).to(device)
    
    vf_estimator = VectorFieldEstimator(
        in_channels=compressed_channels,
        out_channels=compressed_channels,
        hidden_channels=vf_hidden,
        text_dim=vf_text_dim,
        style_dim=vf_style_dim,
        num_style_tokens=se_n_style,
        num_superblocks=vf_n_blocks,
        time_embed_dim=vf_time_dim,
        rope_gamma=float(vf_rotary_scale),
    ).to(device)

    uncond_params = UncondParams(
        text_dim=um_text_dim,
        n_style=um_n_style,
        style_key_dim=um_style_key_dim,
        style_value_dim=um_style_value_dim,
        init_std=uncond_init_std,
    ).to(device)
    u_text = uncond_params.u_text
    u_ref = uncond_params.u_ref
    u_keys = uncond_params.u_keys

    # DP (Optional)
    dp_model = None
    dp_ckpt_path = "checkpoints/duration_predictor/duration_predictor_final.pt"
    if os.path.exists(dp_ckpt_path):
        try:
            print(f"Loading Duration Predictor from {dp_ckpt_path}...")
            dp_model = DPNetwork(vocab_size=37).to(device)
            dp_model.load_state_dict(torch.load(dp_ckpt_path, map_location=device))
            dp_model.eval()
            dp_model.requires_grad_(False)
        except Exception as e:
            print(f"Failed to load DP: {e}")

    # Optimizer
    params = (
        list(text_encoder.parameters()) + 
        list(reference_encoder.parameters()) + 
        list(vf_estimator.parameters()) + 
        list(uncond_params.parameters())
    )
    optimizer = AdamW(params, lr=lr)
    max_steps = 1_000_000
    
    global_step = 0
    
    # Resume
    scheduler_state = None
    ckpts = glob.glob(os.path.join(checkpoint_dir, "ckpt_step_*.pt"))
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_ckpt = ckpts[-1]
        print(f"Found checkpoint: {latest_ckpt}. Resuming...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        
        shapes_changed = False
        if 'vf_estimator' in checkpoint:
            # Filter out size mismatches
            model_state = vf_estimator.state_dict()
            ckpt_state = checkpoint['vf_estimator']
            filtered_state = {}
            for k, v in ckpt_state.items():
                if k in model_state:
                    if v.shape != model_state[k].shape:
                        print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                        shapes_changed = True
                        continue
                    filtered_state[k] = v
            vf_estimator.load_state_dict(filtered_state, strict=False)
        if 'text_encoder' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder'], strict=False)
        if 'reference_encoder' in checkpoint:
            reference_encoder.load_state_dict(checkpoint['reference_encoder'], strict=False)
        if 'u_text' in checkpoint:
            u_text.data = checkpoint['u_text']
        if 'u_ref' in checkpoint:
            u_ref.data = checkpoint['u_ref']
        if 'u_keys' in checkpoint:
            u_keys.data = checkpoint['u_keys']
        
        # Initialize Optimizer here to load state
        optimizer = AdamW(params, lr=lr)

        if 'optimizer' in checkpoint:
            if finetune:
                print(f"Finetune mode: Skipping optimizer state load to use fresh lr={lr}")
            elif shapes_changed:
                print("Warning: Model shapes changed. Skipping optimizer state load to avoid runtime errors.")
            else:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception as e:
                    print(f"Warning: Failed to load optimizer state: {e}. Continuing with fresh optimizer.")
        
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            if finetune:
                spfm_start_override = global_step + spfm_start_override
                print(f"Finetune mode: global_step halved to {global_step} and spfm_start_override set to {spfm_start_override}")


        if 'scheduler' in checkpoint and not shapes_changed:
            scheduler_state = checkpoint['scheduler']
            
        print(f"Resuming from Step {global_step}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # For finetune: start fresh scheduler from -1 (base lr)
    # For resume: continue from current global_step
    scheduler_last_epoch = -1 if finetune else (global_step - 1)
    if scheduler_last_epoch != -1:
        for pg in optimizer.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[300_000, 600_000],
        gamma=0.5,
        last_epoch=scheduler_last_epoch
    )
    
    if scheduler_state is not None and not finetune:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception as e:
            print(f"Warning: Failed to load scheduler state: {e}")

    # DDP Wrapping
    if dist.is_initialized():
        text_encoder = DDP(text_encoder, device_ids=[local_rank], find_unused_parameters=True)
        reference_encoder = DDP(reference_encoder, device_ids=[local_rank], find_unused_parameters=True)
        vf_estimator = DDP(vf_estimator, device_ids=[local_rank], find_unused_parameters=True)
        uncond_params = DDP(uncond_params, device_ids=[local_rank], find_unused_parameters=True)

    # Dataset
    metadata_path = "generated_audio/combined_dataset_cleaned_real_data.csv"
    dataset = Text2LatentDataset(
        metadata_path,
        sample_rate=ae_sample_rate,
        max_wav_len=ae_sample_rate * 20,
        max_text_len=300,
        cross_ref_prob=0.0,  # 50% cross-ref for zero-shot speaker generalization
    )
    if rank == 0:
        print(f"Dataset loaded with {len(dataset)} samples.")

    # Sampler Setup
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        speaker_ids = dataset.speaker_ids
        unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
        freq = dict(zip(unique_speakers, counts))
        print(f"Speaker counts: {freq}")
        
        sample_weights = np.array([1.0 / freq[sid] for sid in speaker_ids])
        sample_weights = sample_weights / sample_weights.sum()
        weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        collate_fn=collate_text2latent,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )
    
    try:
        val_batch = next(iter(dataloader))
        if len(val_batch) == 9:
            val_wavs, val_text_ids, val_text_masks, _, _, val_ref_wavs, val_ref_lengths, val_is_self, _ = val_batch
        else:
             val_wavs, val_text_ids, val_text_masks, _, _, val_ref_wavs, val_ref_lengths, val_is_self = val_batch

        val_wavs = val_wavs[:4].to(device)
        val_text_ids = val_text_ids[:4].to(device)
        val_text_masks = val_text_masks[:4].to(device)
        val_ref_wavs = val_ref_wavs[:4].to(device)
        val_ref_lengths = val_ref_lengths[:4].to(device)
        val_is_self = val_is_self[:4].to(device)
        
        with torch.no_grad():
            val_mel = mel_spec(val_wavs.squeeze(1))
            val_z = ae_encoder(val_mel)
            val_z = compress_latents(val_z, factor=chunk_compress_factor)
            val_z_1 = ((val_z - mean) / std) * normalizer_scale

            val_mel_ref = mel_spec(val_ref_wavs.squeeze(1))
            val_z_ref_full_enc = ae_encoder(val_mel_ref)
            val_z_ref_full_enc = compress_latents(val_z_ref_full_enc, factor=chunk_compress_factor)
            val_z_ref_full = ((val_z_ref_full_enc - mean) / std) * normalizer_scale
            valid_mel_len_ref = val_ref_lengths[:4].to(device).float() / ae_hop_length
            valid_z_len_ref = (valid_mel_len_ref / chunk_compress_factor).ceil().long().clamp(min=1, max=val_z_ref_full.shape[2])
            val_z_ref, val_ref_enc_mask = build_reference_only(val_z_ref_full, valid_z_len_ref, device)
            
    except Exception as e:
        if rank == 0:
            print(f"Validation batch init failed: {e}")
        val_batch = None

    if rank == 0:
        print("Starting training loop...")
    
    epoch = 0
    while global_step < max_steps:
        if dist.is_initialized():
            sampler.set_epoch(epoch)
        epoch += 1
        
        text_encoder.train()
        reference_encoder.train()
        vf_estimator.train()
        
        mode_tag = "[FT] " if finetune else ""
        progress_bar = tqdm(dataloader, desc=f"{mode_tag}Step {global_step}")
        
        epoch_loss = 0.0
        num_batches = 0
        spfm_dirty_total = 0
        spfm_total_samples = 0
        spfm_score_sum = 0.0
        spfm_call_batches = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            if global_step >= max_steps: break
            if len(batch) == 9:
                wavs, text_ids, text_masks, lengths, speaker_ids, ref_wavs, ref_lengths, is_self_ref, ref_speaker_ids = batch
                ref_speaker_ids = ref_speaker_ids.to(device)
            else:
                wavs, text_ids, text_masks, lengths, speaker_ids, ref_wavs, ref_lengths, is_self_ref = batch
                ref_speaker_ids = speaker_ids

            wavs = wavs.to(device)
            text_ids = text_ids.to(device)
            text_masks = text_masks.to(device)
            ref_wavs = ref_wavs.to(device)
            ref_lengths = ref_lengths.to(device)
            is_self_ref = is_self_ref.to(device)
            speaker_ids = speaker_ids.to(device)
            
            if global_step % 100 == 0:
                same_speaker = (speaker_ids == ref_speaker_ids).float().mean().item()
                self_ref_ratio = is_self_ref.float().mean().item()
                if same_speaker < 0.99:
                     print(f"WARNING: Speaker Mismatch! Same-speaker ratio: {same_speaker:.2f}")
                if global_step % 1000 == 0:
                     cross_ref_ratio = 1.0 - self_ref_ratio
                     print(f"[Ref Check] Step {global_step} | Self-Ref: {self_ref_ratio:.2f} | Cross-Ref: {cross_ref_ratio:.2f} | Same-Spk: {same_speaker:.2f}")

            B = wavs.shape[0]

            with torch.no_grad():
                mel = mel_spec(wavs.squeeze(1))
                z = ae_encoder(mel)
                z = compress_latents(z, factor=chunk_compress_factor)
                z_1 = ((z - mean) / std) * normalizer_scale

                mel_ref = mel_spec(ref_wavs.squeeze(1))
                z_ref_full_enc = ae_encoder(mel_ref)
                z_ref_full_enc = compress_latents(z_ref_full_enc, factor=chunk_compress_factor)
                z_ref_full = ((z_ref_full_enc - mean) / std) * normalizer_scale
            
            B, C, T = z_1.shape
            valid_mel_len = lengths.to(device).float() / ae_hop_length
            valid_z_len = (valid_mel_len / chunk_compress_factor).ceil().long().clamp(min=1, max=T)
            latent_mask = (torch.arange(T, device=device).expand(B, T) < valid_z_len.unsqueeze(1)).unsqueeze(1).float()
            z_1 = z_1 * latent_mask

            valid_mel_len_ref = ref_lengths.to(device).float() / ae_hop_length
            valid_z_len_ref = (valid_mel_len_ref / chunk_compress_factor).ceil().long().clamp(min=1, max=z_ref_full.shape[2])
            T_ref_in = z_ref_full.shape[2]
            ref_full_mask = (torch.arange(T_ref_in, device=device).expand(B, T_ref_in) < valid_z_len_ref.unsqueeze(1)).unsqueeze(1).float()
            z_ref_full = z_ref_full * ref_full_mask

            z_ref, ref_enc_mask, _, target_loss_mask = build_reference_from_latents(
                z_1, valid_z_len, z_ref_full, valid_z_len_ref, is_self_ref, device,
                chunk_compress_factor=chunk_compress_factor,
                hop_length=ae_hop_length,
                sample_rate=ae_sample_rate,
            )

            ref_values, _ = reference_encoder(z_ref, mask=ref_enc_mask)

            if global_step % 1000 == 0:
                with torch.no_grad():
                     z_ref_noise = z_ref + (1.0 - ref_enc_mask) * torch.randn_like(z_ref) * 10.0
                     ref_vals_noise, _ = reference_encoder(z_ref_noise, mask=ref_enc_mask)
                     diff = (ref_vals_noise - ref_values).abs().max().item()
                     if diff > 1e-5:
                         print(f"WARNING: ReferenceEncoder is sensitive to padded values! Max Diff: {diff}")

            h_text = text_encoder(text_ids, ref_values, text_mask=text_masks)

            if isinstance(text_encoder, DDP):
                ref_keys = text_encoder.module.ref_keys
            else:
                ref_keys = text_encoder.ref_keys
            ref_keys = ref_keys.expand(B, -1, -1)

            _, D_text, T_txt = h_text.shape

            # SPFM: Self-Purifying Flow Matching — arxiv.org/pdf/2512.17293
            spfm_mask = torch.ones(B, 1, 1, device=device)
            spfm_start = spfm_start_override if spfm_start_override is not None else 40_000

            if global_step >= spfm_start:
                with torch.no_grad():
                    t_spfm = torch.full((B,), 0.5, device=device)
                    t_b = t_spfm.view(B, 1, 1)
                    x0 = torch.randn(B, C, T, device=device)
                    x_t = (1 - (1 - sigma_min) * t_b) * x0 + t_b * z_1
                    v_target_spfm = z_1 - (1 - sigma_min) * x0
                    x_t_in = x_t * latent_mask

                    v_cond = vf_estimator(
                        noisy_latent=x_t_in,
                        text_emb=h_text,
                        style_ttl=ref_values,
                        style_keys=ref_keys,
                        latent_mask=latent_mask,
                        text_mask=text_masks,
                        current_step=t_spfm,
                    )

                    u_text_spfm = u_text.expand(B, D_text, 1)
                    u_text_mask_spfm = torch.ones(B, 1, 1, device=device)

                    v_uncond = vf_estimator(
                        noisy_latent=x_t_in,
                        text_emb=u_text_spfm,
                        style_ttl=u_ref.expand(B, -1, -1),
                        style_keys=u_keys.expand(B, -1, -1),
                        latent_mask=latent_mask,
                        text_mask=u_text_mask_spfm,
                        current_step=t_spfm,
                    )

                    final_mask_spfm = latent_mask * target_loss_mask
                    mask_ct = final_mask_spfm.expand(-1, C, -1)
                    denom = (final_mask_spfm.sum(dim=(1,2)) * C).clamp_min(1)

                    L_cond   = ((v_cond   - v_target_spfm).pow(2) * mask_ct).sum(dim=(1,2)) / denom
                    L_uncond = ((v_uncond - v_target_spfm).pow(2) * mask_ct).sum(dim=(1,2)) / denom
                    spfm_score = L_cond - L_uncond

                    dirty_indices = torch.where(L_cond > L_uncond)[0]
                    if dirty_indices.numel() > 0:
                         spfm_mask[dirty_indices] = 0.0

                    if global_step % 1000 == 0:
                        print(f"[SPFM] Detected Dirty: {dirty_indices.numel()}/{B}")

                dirty_bool = (spfm_mask.squeeze(-1).squeeze(-1) < 0.5)
                dirty_count = dirty_bool.sum().item()
                spfm_dirty_total += dirty_count
                spfm_total_samples += B
                spfm_score_sum += spfm_score.mean().item()
                spfm_call_batches += 1

                if global_step % 1000 == 0 and rank == 0:
                    clean_bool = ~dirty_bool
                    txt_len = text_masks.sum(dim=(1, 2)).float()
                    lat_len = valid_z_len.float()
                    avg_txt_clean = txt_len[clean_bool].mean().item() if clean_bool.any() else 0.0
                    avg_txt_dirty = txt_len[dirty_bool].mean().item() if dirty_bool.any() else 0.0
                    avg_lat_clean = lat_len[clean_bool].mean().item() if clean_bool.any() else 0.0
                    avg_lat_dirty = lat_len[dirty_bool].mean().item() if dirty_bool.any() else 0.0
                    print(
                        f"\n[SPFM Diag] Step {global_step} | Dirty: {dirty_count}/{B} ({dirty_count/B:.1%}) | "
                        f"Score mean: {spfm_score.mean().item():.3f} | "
                        f"TxtLen clean/dirty: {avg_txt_clean:.1f}/{avg_txt_dirty:.1f} | "
                        f"LatLen clean/dirty: {avg_lat_clean:.1f}/{avg_lat_dirty:.1f}"
                    )

            z_1_exp = z_1.repeat_interleave(Ke, dim=0)
            h_text_exp = h_text.repeat_interleave(Ke, dim=0)
            ref_values_exp = ref_values.repeat_interleave(Ke, dim=0)
            ref_keys_exp = ref_keys.repeat_interleave(Ke, dim=0)
            text_masks_base_exp = text_masks.repeat_interleave(Ke, dim=0)
            latent_mask_exp = latent_mask.repeat_interleave(Ke, dim=0)
            target_loss_mask_exp = target_loss_mask.repeat_interleave(Ke, dim=0)
            spfm_mask_exp = spfm_mask.repeat_interleave(Ke, dim=0)

            B_eff = B * Ke

            t = torch.rand(B_eff, device=device)
            with torch.no_grad():
                x_0 = torch.randn(B_eff, C, T, device=device)
                t_broad = t.view(B_eff, 1, 1)
                x_t = (1 - (1 - sigma_min) * t_broad) * x_0 + t_broad * z_1_exp
                v_target = z_1_exp - (1 - sigma_min) * x_0

            # CFG dropout + SPFM dirty routing
            cfg_rand = torch.rand(B_eff, device=device)
            drop_both = cfg_rand < prob_both_uncond
            drop_text_only = (cfg_rand >= prob_both_uncond) & (cfg_rand < puncond)
            force_text_uncond = drop_both | drop_text_only
            force_style_uncond = drop_both.clone()

            is_dirty = (spfm_mask_exp.view(B_eff) < 0.5)
            force_text_uncond = force_text_uncond | is_dirty
            force_style_uncond = force_style_uncond | is_dirty

            mask_text_uncond = force_text_uncond.view(-1, 1, 1).float()
            mask_text_cond = 1.0 - mask_text_uncond
            mask_style_uncond = force_style_uncond.view(-1, 1, 1).float()
            mask_style_cond = 1.0 - mask_style_uncond

            u_text_padded = F.pad(u_text, (0, T_txt - 1))
            h_context = h_text_exp * mask_text_cond + u_text_padded.expand(B_eff, -1, -1) * mask_text_uncond

            mask_uncond_valid = torch.zeros_like(text_masks_base_exp)
            mask_uncond_valid[:, :, 0] = 1.0
            text_mask_final = text_masks_base_exp * mask_text_cond + mask_uncond_valid * mask_text_uncond

            ref_values_final = ref_values_exp * mask_style_cond + u_ref.expand(B_eff, -1, -1) * mask_style_uncond
            ref_keys_final = ref_keys_exp * mask_style_cond + u_keys.expand(B_eff, -1, -1) * mask_style_uncond

            x_t_in = x_t * latent_mask_exp

            # 3. Single Forward Pass
            v_pred = vf_estimator(
                noisy_latent=x_t_in,
                text_emb=h_context,
                style_ttl=ref_values_final,
                style_keys=ref_keys_final,
                latent_mask=latent_mask_exp,
                text_mask=text_mask_final,
                current_step=t
            )
            
            final_mask = latent_mask_exp * target_loss_mask_exp
            loss_raw = F.l1_loss(v_pred, v_target, reduction='none')
            mask_ct = final_mask.expand(-1, C, -1)
            loss = (loss_raw * mask_ct).sum() / (mask_ct.sum() + 1e-8)
            loss = loss / accumulation_steps

            if global_step % 1000 == 0 and rank == 0:
                with torch.no_grad():
                    dirty_rate = (spfm_mask_exp < 0.5).float().mean().item()
                    print(
                        f"\nStep {global_step} Debug:",
                        f"z1 std: {z_1_exp.std().item():.3f}",
                        f"x0 std: {x_0.std().item():.3f}",
                        f"v_target std: {v_target.std().item():.3f}",
                        f"v_pred std: {v_pred.std().item():.3f}",
                        f"\nfinal_mask_mean: {final_mask.mean().item():.3f}",
                        f"dirty_rate: {dirty_rate:.3f}",
                        f"eff_text_uncond: {force_text_uncond.float().mean().item():.3f}",
                        f"eff_style_uncond: {force_style_uncond.float().mean().item():.3f}"
                    )

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            postfix = dict(loss=avg_loss, step=global_step, lr=scheduler.get_last_lr()[0])
            if finetune:
                postfix["mode"] = "FT"
            progress_bar.set_postfix(**postfix)
            
            if global_step % 1000 == 0 and rank == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{global_step}.pt")
                
                vf_state = vf_estimator.module.state_dict() if isinstance(vf_estimator, DDP) else vf_estimator.state_dict()
                te_state = text_encoder.module.state_dict() if isinstance(text_encoder, DDP) else text_encoder.state_dict()
                re_state = reference_encoder.module.state_dict() if isinstance(reference_encoder, DDP) else reference_encoder.state_dict()

                torch.save({
                    'vf_estimator': vf_state,
                    'text_encoder': te_state,
                    'reference_encoder': re_state,
                    'u_text': u_text.data,
                    'u_ref': u_ref.data,
                    'u_keys': u_keys.data,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'global_step': global_step
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                print("Running Inference...")
                vf_infer = vf_estimator.module if isinstance(vf_estimator, DDP) else vf_estimator
                te_infer = text_encoder.module if isinstance(text_encoder, DDP) else text_encoder
                re_infer = reference_encoder.module if isinstance(reference_encoder, DDP) else reference_encoder
                
                vf_infer.eval()
                te_infer.eval()
                re_infer.eval()
                
                try:
                    wav_preds = sample_audio(
                        vf_infer,
                        te_infer,
                        re_infer,
                        ae_decoder,
                        val_text_ids,
                        val_text_masks,
                        val_z_ref,
                        val_ref_enc_mask,
                        u_text, u_ref, u_keys,
                        mean, std,
                        duration_predictor=dp_model,
                        steps=32,
                        cfg_scale=1.2,
                        device=device,
                        latent_dim=latent_dim,
                        chunk_compress_factor=chunk_compress_factor,
                        normalizer_scale=normalizer_scale,
                        hop_length=ae_hop_length,
                    )
                    for idx, wav in enumerate(wav_preds):
                        wav = wav.squeeze().cpu().numpy()
                        sf.write(os.path.join(log_dir, f"step_{global_step}_sample_{idx}.wav"), wav, ae_sample_rate)
                    
                    hebrew_sentences = [
                        "ʃalˈom jaʔakˈov kˈaχa niʃmˈa hamˈodel heχadˈaʃ mˈa daʔtχˈa ? lifʔamˈim tsaʁˈiχ baχajˈim lelatˈeʃ ʁaʔjˈon ʃˈuv vaʃˈuv ʔˈad ʃehˈu matslˈiaχ"
                    ]

                    def run_inference_for_ref(ref_wav_torch, suffix):
                        if ref_wav_torch is None:
                            return

                        with torch.no_grad():
                            ref_mel = mel_spec(ref_wav_torch)
                            ref_z_enc = ae_encoder(ref_mel)
                            ref_z_enc = compress_latents(ref_z_enc, factor=chunk_compress_factor)
                            ref_z_norm = ((ref_z_enc - mean) / std) * normalizer_scale
                            
                            _, _, T_ref = ref_z_norm.shape
                            valid_z_len_ref = torch.tensor([T_ref], device=device)
                            ref_z, ref_mask = build_reference_only(
                                ref_z_norm,
                                valid_z_len_ref,
                                device
                            )

                        for i, text in enumerate(hebrew_sentences):
                            ids = text_to_indices(text)
                            heb_text_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                            heb_text_mask = torch.ones(1, 1, heb_text_ids.shape[1], device=device)
                            
                            wav_out = sample_audio(
                                vf_infer,
                                te_infer,
                                re_infer,
                                ae_decoder,
                                heb_text_ids,
                                heb_text_mask,
                                ref_z,
                                ref_mask,
                                u_text, u_ref, u_keys,
                                mean, std,
                                duration_predictor=dp_model,
                                steps=32,
                                cfg_scale=1.2,
                                device=device,
                                debug_label=suffix,
                                latent_dim=latent_dim,
                                chunk_compress_factor=chunk_compress_factor,
                                normalizer_scale=normalizer_scale,
                                hop_length=ae_hop_length,
                            )
                            wav = wav_out.squeeze().cpu().numpy()
                            sf.write(os.path.join(log_dir, f"step_{global_step}_hebrew_{i+1}_{suffix}.wav"), wav, ae_sample_rate)

                    if 'ref_wav_torch_v1' in locals():
                        run_inference_for_ref(ref_wav_torch_v1, "voice1")

                    if val_batch is not None:
                        ref_z = val_z_ref[0:1]
                        ref_mask = val_ref_enc_mask[0:1]
                        for i, text in enumerate(hebrew_sentences):
                            ids = text_to_indices(text)
                            heb_text_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                            heb_text_mask = torch.ones(1, 1, heb_text_ids.shape[1], device=device)
                            
                            wav_out = sample_audio(
                                vf_infer,
                                te_infer,
                                re_infer,
                                ae_decoder,
                                heb_text_ids,
                                heb_text_mask,
                                ref_z,
                                ref_mask,
                                u_text, u_ref, u_keys,
                                mean, std,
                                duration_predictor=dp_model,
                                steps=32,
                                cfg_scale=1.2,
                                device=device,
                                debug_label="val_sample",
                                latent_dim=latent_dim,
                                chunk_compress_factor=chunk_compress_factor,
                                normalizer_scale=normalizer_scale,
                                hop_length=ae_hop_length,
                            )
                            wav = wav_out.squeeze().cpu().numpy()
                            sf.write(os.path.join(log_dir, f"step_{global_step}_hebrew_{i+1}_val_sample.wav"), wav, ae_sample_rate)
                except Exception as e:
                    print(f"Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                vf_estimator.train()
                text_encoder.train()
                reference_encoder.train()

        if spfm_call_batches > 0 and rank == 0:
            epoch_dirty_rate = spfm_dirty_total / max(spfm_total_samples, 1)
            epoch_score_mean = spfm_score_sum / spfm_call_batches
            print(
                f"[SPFM Epoch] step={global_step} "
                f"dirty_rate={epoch_dirty_rate:.3f} "
                f"score_mean={epoch_score_mean:.3f} "
                f"batches={spfm_call_batches}"
            )

    if rank == 0:
        print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", 
                        help="Finetune mode: lr=5e-4, SPFM starts after warm-up")
    parser.add_argument("--config", type=str, default="configs/tts.json",
                        help="Path to tts.json config file (default: configs/tts.json)")
    parser.add_argument("--Ke", type=int, default=None,
                        help="Override batch expansion factor (default: from config)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 2)")
    args = parser.parse_args()
    
    set_seed(42)
    train(finetune=args.finetune, config_path=args.config, Ke=args.Ke, accumulation_steps=args.accumulation_steps)