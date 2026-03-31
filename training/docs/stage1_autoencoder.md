# Stage 1: Speech Autoencoder

**Script:** `src/train_autoencoder.py`
**Hardware:** 2× RTX 3090 (PyTorch DDP)
**Iterations:** 1,500,000

The Speech Autoencoder compresses 44.1 kHz audio into a compact 24-dimensional continuous latent representation at ~86 Hz, then reconstructs the waveform from that latent. It is trained with a combination of reconstruction, adversarial, and feature matching losses.

---

## Architecture

```
 Input Audio [B, 1, T]   (44.1 kHz waveform)
        │
        │  LinearMelSpectrogram (n_fft=2048, hop=512, n_mels=228)
        │  + log-linear (1025 bins)  →  concatenate
        ▼
 Spectrogram [B, 1253, T/512]
 (1025 log-linear bins + 228 log-mel bins)
        │
        ▼
 ┌────────────────────────── ENCODER (~25.6M) ──────────────────────────┐
 │                                                                       │
 │  Conv1d stem  1253 → 512   (kernel=7, BatchNorm)                     │
 │        │                                                              │
 │  ┌─────┴──────────────────────────────────────────────────────────┐  │
 │  │  ConvNeXt Block × 10   (dilations all 1)                       │  │
 │  │                                                                 │  │
 │  │  Each block:                                                    │  │
 │  │    DW-Conv1d (kernel=7) ──► LayerNorm ──► PW-Conv 512→2048     │  │
 │  │    ──► GELU ──► PW-Conv 2048→512 ──► γ-scale ──► residual add  │  │
 │  └─────────────────────────────────────────────────────────────────┘  │
 │        │                                                              │
 │  Conv1d proj  512 → 24  (1×1)  +  LayerNorm                         │
 └───────────────────────────────────────────────────────────────────────┘
        │
        ▼
   Latent z [B, 24, T/512]   (~86 Hz)
        │
        ▼
 ┌────────────────────────── DECODER (~25.3M) ──────────────────────────┐
 │                                                                       │
 │  CausalConv1d stem  24 → 512   (kernel=7)                            │
 │        │                                                              │
 │  ┌─────┴──────────────────────────────────────────────────────────┐  │
 │  │  Causal ConvNeXt Block × 10  (dilations: 1,2,4,1,2,4,1,1,1,1) │  │
 │  │                                                                 │  │
 │  │  Same structure as encoder but CAUSAL (left-pad only),         │  │
 │  │  enabling streaming/real-time inference.                       │  │
 │  └─────────────────────────────────────────────────────────────────┘  │
 │        │                                                              │
 │  BatchNorm  ──►  VocoderHead                                         │
 │                                                                       │
 │  VocoderHead:                                                         │
 │    CausalConv1d  512→2048  (kernel=3)                                │
 │    ──► PReLU                                                          │
 │    ──► Conv1d 2048→512  (1×1)                                        │
 │    ──► reshape [B, 512, T] → [B, T*512]   (sub-pixel expansion)     │
 └───────────────────────────────────────────────────────────────────────┘
        │
        ▼
   Reconstructed Waveform [B, T_audio]
```

### ConvNeXt Block (detailed)

```
   x [B, C, T]
        │
        ├── residual ─────────────────────────────────────────┐
        │                                                      │
        │  DW-Conv1d  (groups=C, kernel=7, dilation=d)        │
        │        │                                             │
        │  LayerNorm (channel-wise)                           │
        │        │                                             │
        │  PW-Conv  C → intermediate (2048)                   │
        │        │                                             │
        │       GELU                                           │
        │        │                                             │
        │  PW-Conv  2048 → C                                  │
        │        │                                             │
        │   × γ  (learnable scalar, init=1e-6)                │
        │        │                                             │
        └────────┤ + ◄─────────────────────────────────────────┘
                 │
              output
```

### Causal vs Non-Causal Convolution

```
  Non-causal (Encoder):
  ← · · · [t-3][t-2][t-1][ t ][t+1][t+2][t+3] · · · →
                      └────── kernel ──────┘
           padding = (kernel-1)*dilation // 2  (both sides)

  Causal (Decoder):
  ← · · · [t-3][t-2][t-1][ t ]
                      └── kernel ──┘
           padding = (kernel-1)*dilation  (left side only)
           → no future context, suitable for streaming
```

---

## Discriminators (GAN Training)

Two discriminators run in parallel to provide adversarial signal.

### Multi-Period Discriminator (MPD)

```
  y_hat (waveform)
        │
  ┌─────┴──────────────────────────────────────────────────────────┐
  │  Period 2  │  Period 3  │  Period 5  │  Period 7  │  Period 11  │
  │            │            │            │            │             │
  │  Reshape waveform to 2D: [B, 1, T/p, p]                        │
  │  Stack of Conv2d layers with stride → scalar score             │
  └──────────────────────────────────────────────────────────────────┘
        │
  scores_real, scores_fake, fmaps_real, fmaps_fake
```

### Multi-Resolution Discriminator (MRD)

```
  y_hat (waveform)
        │
  ┌───────────────────────────────────────────────┐
  │  FFT 512  │  FFT 1024  │  FFT 2048            │
  │           │            │                      │
  │  Compute magnitude spectrogram per resolution │
  │  Stack of Conv2d layers → scalar score        │
  └───────────────────────────────────────────────┘
        │
  scores_real, scores_fake, fmaps_real, fmaps_fake
```

---

## Loss Functions

### Generator Loss (L_G)

```
L_G = 45 · L_recon  +  1 · L_adv  +  0.1 · L_fm
```

**Reconstruction Loss (L_recon)** — multi-resolution mel L1, averaged over 3 scales:

| Scale | n_fft | hop | win | n_mels |
|---|---|---|---|---|
| Small | 1024 | 256 | 1024 | 64 |
| Medium | 2048 | 512 | 2048 | 128 |
| Large | 4096 | 1024 | 4096 | 128 |

```
L_recon = (1/3) Σ_scale  L1( Mel_real(crop), Mel_fake(crop) )
```

**Adversarial Loss (L_adv)** — LSGAN generator objective:

```
L_adv = Σ_D  E[ (1 - D(G(x)))² ]
```

**Feature Matching Loss (L_fm)** — L1 distance between intermediate discriminator features:

```
L_fm = (1/N) Σ_{layers}  E[ |feat_real - feat_fake| ]
```

### Discriminator Loss (L_D)

LSGAN discriminator objective (real → 1, fake → −1):

```
L_D = Σ_D  E[ (D(x) - 1)²  +  (D(G(x)) + 1)² ]
```

---

## Training Configuration

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.8, β₂=0.99, wd=0.01) |
| Learning rate | 2e-4 |
| LR schedule | CosineAnnealingLR → 1e-6 over 1.5M steps |
| Batch size | 128 |
| Audio crop length | 0.19 s (~8,379 samples @ 44.1 kHz) |
| Total iterations | 1,500,000 |
| Grad clip (encoder/decoder) | 5.0 |
| Grad clip (discriminators) | 1.0 |
| Discriminator warmup | 0 steps (active immediately) |

### Training Step Flow

```
 batch (raw audio segments)
        │
        ▼
 1. Compute spectrogram (no_grad)
        │
        ▼
 2. Encoder → z → Decoder → y_hat
        │
        ▼
 3. Random crop of audio & y_hat (0.19 s window)
        │
        ├────────────────────────────────────┐
        ▼                                    ▼
 4. Discriminator step                5. Generator step
    (y_hat detached)                      (y_hat with grads)
        │                                    │
    MPD + MRD → L_D                    MPD + MRD → L_adv
    backward + clip                    + L_fm + L_recon
    opt_d.step()                       → L_G  backward + clip
                                       opt_g.step()
        │                                    │
        └──────────────── step ──────────────┘
                          │
                    scheduler_g.step()
                    scheduler_d.step()
```

### Checkpoint Layout

```
checkpoints/ae/
├── ae_latest.pt          ← always points to most recent
├── ae_1000.pt
├── ae_2000.pt
├── ae_3000.pt            ← rolling window (last 1000 kept)
└── logs/                 ← TensorBoard events (auto-started port 8000)
```

Checkpoint contents:

```python
{
    "step": int,
    "epoch": int,
    "encoder": state_dict,
    "decoder": state_dict,
    "mpd": state_dict,
    "mrd": state_dict,
    "opt_g": state_dict,
    "opt_d": state_dict,
    "scheduler_g": state_dict,
    "scheduler_d": state_dict,
}
```

---

## Running Stage 1

### Initial training

```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py \
    --arch_config configs/tts.json
```

### Resume from checkpoint

```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py \
    --resume checkpoints/ae/ae_latest.pt
```

### Finetune (reset optimizer + step counter)

```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py \
    --resume checkpoints/ae/ae_latest.pt \
    --finetune \
    --lr 1e-5
```

### With reconstruction evaluation sample

```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py \
    --eval_input path/to/sample.wav
```

---

## Next Step

After Stage 1 completes (or at any checkpoint), run:

```bash
python compute_latent_stats.py --tts-json configs/tts.json
# → outputs: stats_real_data.pt
```

This is **required** before running Stage 2 or Stage 3.
