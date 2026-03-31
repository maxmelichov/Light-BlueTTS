# Stage 3: Duration Predictor

**Script:** `src/train_duration_predictor.py`
**Hardware:** 1× GPU
**Iterations:** 3,000–6,000
**Prerequisites:** Stage 1 complete + `stats_real_data.pt`

The Duration Predictor (DP) estimates the **total number of compressed-latent frames** that a given utterance will occupy, conditioned on a short reference speech clip. It operates at the **utterance level** (not phoneme-level), predicting a single scalar duration per utterance.

At inference, this duration is used to set the target length for the flow-matching model before generation begins.

---

## Why Utterance-Level?

Unlike phoneme-level duration models (which require phoneme alignments), this model predicts total duration from:
- The **text** content (how many characters/words)
- A **reference speech** sample from the target speaker (captures speaking rate, style)

This avoids the need for forced alignment tools and works well with character-level tokenization.

---

## Architecture (~0.5M parameters)

```
 Text string                         Reference audio clip
      │                                      │
      │ char → IDs                           │ AE Encoder (frozen)
      ▼                                      ▼
 ┌──────────────────────────┐    ┌────────────────────────────────────┐
 │   DP Text Encoder        │    │   DP Reference Encoder             │
 │   (~0.35M)               │    │   (~0.12M)                         │
 │                          │    │                                    │
 │  CharEmb (vocab=37,      │    │  z_ref [B, 144, T_ref]             │
 │           dim=64)        │    │       │                            │
 │       │                  │    │  Conv1d 144 → 64  (1×1)            │
 │  prepend utterance       │    │       │                            │
 │  token [B, 64, 1]        │    │  ConvNeXt × 4 (64-dim, expand=4)  │
 │       │                  │    │       │ (as Key/Value)             │
 │  ConvNeXt × 6            │    │                                    │
 │  (64-dim, expand=4)      │    │  Learnable Queries [8, 16]         │
 │       │                  │    │       │                            │
 │  Self-Attn × 2           │    │  Cross-Attention × 2  (1 head)    │
 │  (2 heads, filter=256)   │    │       │                            │
 │       │                  │    │  reshape: [B, 8, 16] → [B, 128]   │
 │  + residual (ConvNeXt    │    │                                    │
 │    output)               │    └──────────────── style_emb ─────────┘
 │       │                  │                           │
 │  take first token        │                           │
 │  Proj 64→64 (1×1)        │                           │
 └─────── text_emb ─────────┘                           │
              │                                         │
              └──────────────────┬──────────────────────┘
                                 │
                          concat [B, 192]
                          (64 text + 128 style)
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Duration Estimator     │
                    │                         │
                    │  Linear  192 → 128      │
                    │  PReLU                  │
                    │  Linear  128 → 1        │
                    │                         │
                    │  output: log(duration)  │
                    └─────────────────────────┘
                                 │
                    loss: L1( log_pred, log(z_frames) )
```

---

## Component Details

### DP Text Encoder

```
 text_ids [B, T]
        │
  CharEmbedding  [vocab=37, 64-dim]
        │
  transpose → [B, 64, T]
        │
  prepend utterance token  →  [B, 64, T+1]
        │   (learnable [64, 1] vector, always at position 0)
        │
  ConvNeXt × 6  (dim=64, kernel=5, intermediate=256)
        │ ←── save as conv_out
        │
  Self-Attention × 2  (2 heads, filter_channels=256, LARoPE)
        │
  + conv_out  (residual connection)
        │
  take slice [:, :, 0]  → [B, 64, 1]  ← utterance token
        │
  Proj (Conv1d 1×1, no bias)
        │
  text_emb [B, 64]
```

The utterance token aggregates global context from the full sequence via attention, producing a single 64-dim summary vector.

### DP Reference Encoder

```
 z_ref [B, 144, T_ref]   (normalized compressed latent)
        │
  Conv1d 144→64  (1×1)   → [B, 64, T_ref]
        │
  ConvNeXt × 4  (dim=64, expand=4)
        │
  transpose → [B, T_ref, 64]   ← Key/Value for cross-attention
        │
  Learnable Queries  [8, 16]   ← 8 style tokens of 16 dims each
  (expanded to [B, 8, 16])
        │
  Cross-Attention × 2  (1 head, kdim=vdim=64, embed_dim=16)
  q = q + attn(q, kv, kv)   (residual)
        │
  reshape [B, 8, 16] → [B, 128]   ← style_emb
```

The 8 learnable query vectors act as style summarizers. Each independently attends to the encoded reference audio, producing a compact 128-dim speaker/style embedding.

### Duration Estimator

```
  text_emb  [B,  64]
  style_emb [B, 128]
         │
  cat → [B, 192]
         │
  Linear 192 → 128
         │
  PReLU
         │
  Linear 128 → 1
         │
  log_duration [B]      (predict log, exponentiate at inference)
```

Predicting in log-space ensures the model cannot output negative durations and compresses the heavy tail of the duration distribution.

---

## Reference Segment Sampling

During training, a random crop of each utterance is used as the style reference (5%–95% of total duration):

```
Full utterance latent: z  [T_total frames]
  ├────────────────────────────────────────┤

  start = random in [5% · T_total, 95% · T_total - 1]
  end   = random in [start+1, 95% · T_total]

  z_ref = z[:, :, start:end]   → reference for DP
  target = T_total             → what DP must predict
```

This forces the model to generalize reference crops of varying length and position, making it robust to diverse reference clips at inference.

---

## Loss Function

L1 loss on the log-duration:

```
log_pred = DP(text, z_ref)              (model output in log domain)
log_gt   = log( T_total_latent_frames ) (ground truth)

L_DP = L1( log_pred, log_gt )
```

At inference, convert back:

```
T_frames = exp( log_pred )   ≈ round to nearest integer
```

---

## Speaker Balancing

The dataset contains many speakers with very different numbers of utterances. To prevent dominant speakers from overfitting, **WeightedRandomSampler** is used:

```python
weight[i] = 1.0 / count(speaker[i])
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
```

This gives equal expected gradient contribution per speaker regardless of dataset imbalance.

---

## Training Configuration

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 64–128 |
| Reference crop | 5%–95% of utterance |
| Total iterations | 3,000–6,000 |
| AE encoder | Frozen (no grad) |
| Vocab size | 37 (Hebrew chars + punctuation) |
| Style tokens | 8 |
| Style token dim | 16 |

### Running Stage 3

```bash
python src/train_duration_predictor.py \
    --config configs/tts.json \
    --max_steps 6000 \
    --batch_size 64 \
    --lr 1e-4
```

### Checkpoint layout

```
checkpoints/duration_predictor/
├── duration_predictor_500.pt
├── duration_predictor_1000.pt
├── ...
└── duration_predictor_final.pt   ← saved after max_steps
```

---

## Data Flow During Training

```
  batch (wavs, text_ids, text_masks, wav_lengths, speaker_ids)
         │
         ▼
  1. mel_spec(wavs) → mel [B, 228, T_mel]          (no_grad)
         │
  2. ae_encoder(mel) → z [B, 24, T_lat]            (no_grad, AE frozen)
         │
  3. compress_latents(z, factor=6) → z_c [B, 144, T_lat_c]
         │
  4. normalize: z_c = (z_c - mean) / std
         │
  5. compute valid latent lengths from wav_lengths
         │
  6. sample random reference segment [5%, 95%] of valid length
         │  → z_ref [B, 144, T_ref]
         │
  7. DP forward:
         text_ids → text_emb [B, 64]
         z_ref    → style_emb [B, 128]
         → log_pred [B]
         │
  8. loss = L1(log_pred, log(valid_z_len))
         │
  9. backward + optimizer step
```

---

## Relationship to Stage 2 (TTL)

```
  At inference:
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. DP predicts T_frames = exp(DP(text, ref_audio))             │
  │                                                                  │
  │  2. TTL generates z_c [B, 144, T_frames] via Euler solver       │
  │     starting from z_0 ~ N(0, I) of shape [B, 144, T_frames]     │
  │                                                                  │
  │  3. Decompress: z [B, 24, T_frames × 6]                         │
  │                                                                  │
  │  4. AE Decoder: z → waveform [B, T_frames × 6 × 512 samples]   │
  └──────────────────────────────────────────────────────────────────┘
```

The DP does not affect the quality of the generated speech — only its **duration**. The TTL model can generate speech of any length; the DP provides the target length estimate.
