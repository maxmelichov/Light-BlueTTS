# Training — Light-BlueTTS

Training is split into three independent stages that must be run in order:

1. **Speech Autoencoder** — encodes audio into a 24-dim continuous latent space
2. **Text-to-Latent (TTL)** — flow-matching model that maps text + reference speech to latents
3. **Duration Predictor (DP)** — utterance-level duration estimator

Total inference model size: ~71M parameters (AE decoder ~25M, TTL ~45M, DP ~0.5M).

---

## Directory Structure

```
training/
├── src/
│   ├── train_autoencoder.py        # Stage 1: AE training (multi-GPU DDP)
│   ├── train_text_to_latent.py     # Stage 2: TTL flow-matching training
│   └── train_duration_predictor.py # Stage 3: Duration predictor training
├── models/
│   ├── autoencoder/
│   │   ├── latent_encoder.py       # LatentEncoder (mel → 24-dim latent)
│   │   ├── latent_decoder.py       # LatentDecoder1D (latent → waveform)
│   │   ├── discriminators.py       # MPD + MRD for GAN training
│   │   └── modules.py              # Shared: ConvNeXtBlock, CausalConvNeXtBlock, etc.
│   └── text2latent/
│       ├── text_encoder.py         # TextEncoder with ConvNeXt + self-attn + style cross-attn
│       ├── reference_encoder.py    # ReferenceEncoder (audio → style tokens)
│       ├── vf_estimator.py         # VectorFieldEstimator (flow-matching backbone)
│       ├── duration_predictor.py   # TTSDurationModel (full DP model)
│       └── dp_network.py           # DPNetwork (backward-compatible wrapper)
├── compute_latent_stats.py         # Compute latent mean/std (run before Stage 2/3)
└── configs/
    └── tts.json                    # Single config file for all stages
```

---

## Architecture

### Speech Autoencoder

Encodes 44.1 kHz audio via a concatenated log-linear (1025-ch) + log-mel (228-ch) spectrogram (FFT 2048, hop 512) into a 24-dim latent at ~86 Hz.

| Component | Details |
|---|---|
| Input | 1253-channel spectrogram (1025 log-linear + 228 log-mel) |
| Encoder (~25.6M) | Conv1d stem (1253→512) + 10 ConvNeXt blocks (intermediate 2048) + proj (512→24) |
| Decoder (~25.3M) | CausalConv1d stem (24→512) + 10 causal dilated ConvNeXt blocks + VocoderHead |
| Decoder dilations | `[1, 2, 4, 1, 2, 4, 1, 1, 1, 1]` |
| Discriminators | MPD (periods 2,3,5,7,11) + MRD (FFTs 512/1024/2048) |

**Generator loss:**
```
L_G = 45 * L_recon + 1 * L_adv + 0.1 * L_fm
```
Reconstruction loss is multi-resolution mel L1 on 3 scales: (FFT 1024, 64 mels), (FFT 2048, 128 mels), (FFT 4096, 128 mels).

### Text-to-Latent Module

Operates on *compressed* latents: the 24-dim latent is reshaped to 144-dim at ~14 Hz (compression factor K_c = 6).

| Component | Details |
|---|---|
| Reference Encoder (~4.8M) | Conv1d (144→256) + 6 ConvNeXt blocks (k=5) + 2 cross-attn layers → 50 style tokens |
| Text Encoder (~6.9M) | Char embedding (256-dim) + 6 ConvNeXt blocks + 4 self-attn blocks (RoPE) + 2 style cross-attn layers |
| Vector Field Estimator (~33M) | proj_in (144→512) + 4× superblock + 4 final ConvNeXt blocks + proj_out (512→144) |
| VF superblock | 4× dilated ConvNeXt (d=1,2,4,8) + time injection + 2× ConvNeXt + text cross-attn + style cross-attn |

**Flow-matching objective (L1, masked):**
```
L_TTL = E [ || m · (v(z_t, z_ref, c, t) - (z₁ - (1 - σ_min)·z₀)) ||₁ ]
```
where `m` is a reference mask, σ_min = 1e-8, and `p_uncond = 0.05` for classifier-free guidance.
Inference uses Euler's method with NFE=32 and CFG coefficient 3.

### Duration Predictor

Utterance-level (not phoneme-level). ~0.5M parameters.

| Component | Details |
|---|---|
| DP Reference Encoder | Linear (144→64) + 4 ConvNeXt blocks + 2 cross-attn → 64-dim embedding |
| DP Text Encoder | Char embedding (64-dim) + 6 ConvNeXt blocks + 2 self-attn + utterance token → 64-dim |
| Estimator | Linear(192→128) + PReLU + Linear(128→1) → scalar log-duration |

Loss: L1 on log-duration.

---

## Reducing Model Size

All dimensions are controlled by `config/tts.json`. The main levers:

### Speech Autoencoder (~51M → smaller)

The largest single cost is the encoder's stem layer because `idim=1253` (1025 log-linear + 228 log-mel channels). Switching to pure mel drops that:

| Change | Param reduction |
|---|---|
| `encoder.idim`: 1253 → 228 (mel-only input) | −3.7M |
| `encoder.hdim`: 512 → 256 | −10M (stems + all blocks) |
| `encoder.intermediate_dim`: 2048 → 1024 | −10.5M (all blocks) |
| Reduce `encoder.num_layers` / `decoder.num_layers` 10 → 6 | −8.4M each |

> Note: reducing `idim` to 228 requires also changing `LinearMelSpectrogram` to output only mel (set `n_mels=228` and remove the log-linear concatenation in `models/utils.py`).

### Text-to-Latent (~45M → smaller)

The VF Estimator dominates (~33M). Key knobs in `config/tts.json`:

| Config key | Current | Smaller option | Effect |
|---|---|---|---|
| `vector_field.proj_in.odim` (hidden dim) | 512 | 256 | −24M (all VF blocks scale quadratically) |
| `vector_field.main_blocks.n_blocks` | 4 superblocks | 2 | −14M |
| `vector_field.main_blocks.convnext_0.dilation_lst` | [1,2,4,8] | [1,2] | −5M per superblock |
| `ttl.text_encoder.convnext.num_layers` | 6 | 4 | −1.1M |
| `ttl.text_encoder.attn_encoder.n_layers` | 4 | 2 | −1.6M |

A practical small TTL (~10M): set `proj_in.odim=256`, `n_blocks=2`, reduce convnext to 4 layers.

---

## Training

### Prerequisites

```bash
# Install dependencies
uv sync  # or pip install -e ..

# Prepare your dataset metadata CSV with columns: path, text, speaker_id
# Expected format: generated_audio/combined_dataset_cleaned_real_data.csv
#
# Training dataset: ~5.9M files / ~10,000 hours of audio
```

---

### Stage 1: Speech Autoencoder

Multi-GPU training via PyTorch DDP on 4× GPUs.

```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py \
    --arch_config configs/tts.json
```

**Key hyperparameters** (from paper Sec. 4.1):

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.8, β₂=0.99, wd=0.01) |
| Learning rate | 2e-4 with cosine annealing to 1e-6 |
| Batch size | 128 |
| Crop length | 0.19 s (~8,379 samples at 44.1 kHz) |
| Total iterations | 1,500,000 |
| Hardware (paper) | 2× RTX 3090 |

Resume from checkpoint:
```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py \
    --resume checkpoints/ae/ae_latest.pt
```

Checkpoints are saved to `checkpoints/ae/` every `save_interval` steps (configured in `tts.json`). TensorBoard logs are written to `checkpoints/ae/logs/` and auto-started on port 8000.

---

### Compute Latent Statistics

Must be run **after Stage 1** and **before Stages 2 and 3**. Computes per-channel mean and std over compressed latents for normalization.

```bash
python compute_latent_stats.py --tts-json configs/tts.json
# Outputs: stats_real_data.pt
```

---

### Stage 2: Text-to-Latent

```bash
python src/train_text_to_latent.py --config configs/tts.json
```

**Key hyperparameters** (from paper Sec. 4.2):

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4, halved every 300k iterations |
| Batch size | 64 |
| Expansion factor K_e | 4 |
| σ_min | 1e-8 |
| p_uncond | 0.05 |
| Reference crop | 0.2–9 s, ≤50% of utterance length |
| Total iterations | 700,000 |
| Hardware (paper) | 2× RTX 3090 |

---

### Stage 3: Duration Predictor

Requires a trained AE encoder checkpoint and latent stats.

```bash
python src/train_duration_predictor.py \
    --config configs/tts.json \
    --max_steps 6000 \
    --batch_size 64 \
    --lr 1e-4
```

**Key hyperparameters** (from paper Sec. 4.3):

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 64–128 |
| Reference crop | 5%–95% of utterance |
| Total iterations | 3,000–6,000 |
| Hardware (paper) | 1× RTX 3090 |

The AE encoder is frozen during DP training. Speaker-balanced sampling (`WeightedRandomSampler`) is used to handle imbalanced datasets.

---

## Configuration

All model dimensions and training parameters are controlled by `configs/tts.json`. Key sections:

```json
{
  "ae": {
    "encoder": { "ksz": 7, "hdim": 512, "intermediate_dim": 2048, "odim": 24, ... },
    "decoder": { "ksz": 7, "hdim": 512, "dilation_lst": [1,2,4,1,2,4,1,1,1,1], ... },
    "data": { "sample_rate": 44100, "segment_size": 8379, "batch_size": 128 }
  },
  "ttl": {
    "latent_dim": 24,
    "chunk_compress_factor": 6,
    "batch_expander": { "n_batch_expand": 4 },
    "flow_matching": { "sig_min": 1e-8 },
    ...
  },
  "dp": {
    "style_encoder": { "style_token_layer": { "n_style": 8, "style_value_dim": 16 } }
  }
}
```

---

## Reference

```bibtex
@article{kim2025supertonictts,
  title={SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System},
  author={Kim, Hyeongju and Yang, Jinhyeok and Yu, Yechan and Ji, Seunghun and Morton, Jacob and Bous, Frederik and Byun, Joon and Lee, Juheon},
  journal={arXiv preprint arXiv:2503.23108},
  year={2025}
}
```
