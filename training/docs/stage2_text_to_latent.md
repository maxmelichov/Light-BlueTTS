# Stage 2: Text-to-Latent (TTL)

**Script:** `src/train_text_to_latent.py`
**Hardware:** 2× RTX 3090
**Iterations:** 700,000
**Prerequisite:** Stage 1 complete + `stats_real_data.pt` generated

The Text-to-Latent module is a **flow-matching** generative model. Given an input text and a reference speech clip, it learns to generate the corresponding compressed latent representation. At inference, it uses Euler's method with 32 function evaluations (NFE=32) and classifier-free guidance (CFG=3).

---

## Latent Compression

Before the TTL model operates, the 24-dim AE latent is temporally compressed:

```
z  [B, 24, T]   (AE latent @ ~86 Hz)
        │
        │  reshape: pack K_c=6 consecutive frames → one frame
        ▼
z_c [B, 144, T/6]   (compressed latent @ ~14 Hz)

  Example: T=860 frames → T/6 ≈ 143 compressed frames
           144 = 24 × 6 channels encode spatial+temporal info together
```

This 6× compression reduces the sequence length that the model must generate, making training and inference significantly faster.

---

## Overall Architecture

```
 Text string
      │
      │  char → token IDs  (Hebrew char vocabulary, size=37 + special)
      ▼
 ┌───────────────────────────────── TEXT ENCODER (~6.9M) ─────────────────────────────────┐
 │                                                                                         │
 │  CharEmbedding (256-dim)                                                                │
 │       │                                                                                 │
 │  ConvNeXt × 6  (256-dim, k=5, all dilation=1)                                          │
 │       │                                                                                 │
 │  Self-Attention × 4  (4 heads, LARoPE, filter=1024, p_drop=0.1)                        │
 │       │                                                                                 │
 │  Style Cross-Attention × 2  (query=text, key/value=style tokens)                       │
 │       │                                                                                 │
 │  Proj 256→256  (1×1)                                                                   │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
      │
      ▼ h_text [B, 256, T_text]

 Reference audio clip
      │
      │  AE Encoder (frozen) → z_ref [B, 24, T] → compress → z_ref_c [B, 144, T_ref]
      ▼
 ┌───────────────────────── REFERENCE ENCODER (~4.8M) ─────────────────────────┐
 │                                                                               │
 │  Conv1d  144 → 256  (1×1)                                                    │
 │       │                                                                       │
 │  ConvNeXt × 6  (256-dim, k=5)                                                │
 │       │                                                                       │
 │  Sinusoidal pos embedding added                                               │
 │       │ (as Key/Value sequence)                                               │
 │                                                                               │
 │  Learnable Queries [50, 256]  (ref_keys)                                     │
 │       │ (cross-attend TO audio features)                                      │
 │  Cross-Attention × 2  (4 heads, pre-norm)                                    │
 │       │                                                                       │
 │  → ref_values [B, 50, 256]   (context-aware style summary)                   │
 │  → ref_keys   [B, 50, 256]   (static learnable tokens)                       │
 └───────────────────────────────────────────────────────────────────────────────┘

       ref_values + ref_keys
              │
              ▼
 ┌──────────────────────────────────── VECTOR FIELD ESTIMATOR (~33M) ──────────────────────────────────────┐
 │                                                                                                          │
 │  Input: noisy latent z_t [B, 144, T_lat]   (linear interpolation of z_0 noise and z_1 target)          │
 │                                                                                                          │
 │  proj_in: Conv1d 144 → 512  (1×1, no bias)                                                              │
 │       │                                                                                                  │
 │  Time t [B] → SinusoidalEmb(64) → MLP(64→256→64) → t_emb [B, 64]                                      │
 │                                                                                                          │
 │  ┌──────────────── Superblock × 4 ────────────────────────────────────────────────────────────────────┐ │
 │  │                                                                                                     │ │
 │  │  ConvNeXtStack  (dilations=[1,2,4,8], k=5)   ← receptive field: ±(4+8+16+32)=60 frames            │ │
 │  │       │                                                                                             │ │
 │  │  TimeCondBlock  x += Linear(t_emb → 512)   ← additive time injection                              │ │
 │  │       │                                                                                             │ │
 │  │  ConvNeXtStack  (dilation=[1])                                                                     │ │
 │  │       │                                                                                             │ │
 │  │  TextCrossAttn  (LARoPE, 4 heads, 256-dim)   ← attend to h_text                                   │ │
 │  │       │                                                                                             │ │
 │  │  ConvNeXtStack  (dilation=[1])                                                                     │ │
 │  │       │                                                                                             │ │
 │  │  StyleCrossAttn  (no RoPE, tanh keys, 2 heads, 256-dim) ← attend to ref_values                    │ │
 │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
 │       │                                                                                                  │
 │  last_convnext: ConvNeXtStack  (dilations=[1,1,1,1])                                                    │
 │       │                                                                                                  │
 │  proj_out: Conv1d 512 → 144  (1×1, no bias)                                                             │
 │                                                                                                          │
 │  Output: vector field prediction v̂ [B, 144, T_lat]                                                     │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
       │
       ▼  (Euler step at inference)
 z_1 (predicted latent)  → AE Decoder → Waveform
```

---

## Flow Matching

### Training: Constructing z_t

Flow matching interpolates linearly between a noise sample z_0 and the target latent z_1:

```
z_t = t · z_1  +  (1 - (1 - σ_min) · t) · z_0
```

where σ_min = 1e-8 (near-zero), so effectively:

```
z_t ≈ t · z_1  +  (1 - t) · z_0     for t ∈ [0, 1]

  t=0: pure noise z_0
  t=1: clean latent z_1
```

The model learns to predict the **conditional vector field**:

```
target = z_1 - (1 - σ_min) · z_0  ≈  z_1 - z_0
```

### Training Loss

Masked L1 loss on the predicted vector field:

```
L_TTL = E[ || m ⊙ (v̂(z_t, z_ref, c, t) - (z_1 - z_0)) ||₁ ]

  m  = reference mask (1 where target should be predicted, 0 for reference frames)
  z_t = noisy interpolated latent
  z_ref = compressed reference latent
  c  = conditioned text + style context
  t  ~ Uniform(0, 1)
```

### Classifier-Free Guidance (CFG)

With probability p_uncond = 0.05, the model is trained with conditioning dropped (null text or null style), enabling guidance at inference:

```
v_guided = v_uncond  +  cfg · (v_cond - v_uncond)

  cfg = 3  (inference default)
```

### Inference: Euler Solver

```
z_0 ~ N(0, I)

for step s in [0, NFE-1]:   (NFE = 32)
    t   = s / NFE
    v̂  = VFE(z_t, text, ref, t)
    z_t += (1 / NFE) · v̂       (Euler step)

z_1 = z_T   (decompressed → AE Decoder → waveform)
```

```
  Time →
  z_0 ──► z_1/32 ──► z_2/32 ──► ... ──► z_31/32 ──► z_1
   noise                                             target latent
  │   Δt=1/32  │   Δt=1/32  │              │   Δt=1/32  │
  └──── v̂₀ ────┴──── v̂₁ ────┴─── ... ────┴──── v̂₃₁ ────┘
```

---

## Attention Mechanisms

### Length-Aware RoPE (LARoPE) — Text Attention

Positions are **normalized by sequence length** before applying rotary embeddings. This makes the model position-invariant to absolute length, inducing a relative diagonal attention pattern:

```
pos_normalized[i] = i / len(sequence)

freq[i] = pos_normalized[i] · θ · γ      (γ=10 scaling factor)
cos/sin applied to Q and K before dot-product
```

### Style Attention (no RoPE)

Style tokens are attended to without positional bias. Keys are passed through `tanh` to bound them:

```
K = tanh(W_K · style_values)
attn = softmax(Q · K^T / √256) · V
```

---

## Reference Mask and Self-Reference

During training, each sample uses a **crop of itself** as the style reference:

```
Full utterance:   z_1  [B, 144, T_total]
                  ├──────────────────────┤
                  │    reference crop    │  0.2–9 s, ≤50% of utterance
                  ├──────┤               │
                  0     ref_end        T_total

  - ref_start: random within [0, T_total - min_frames]
  - ref_len:   random in [min_frames, max_frames] ∩ [0, 0.5 · T_total]
  - min_frames ≈ 0.2 s × 14.35 Hz ≈ 3
  - max_frames ≈ 9.0 s × 14.35 Hz ≈ 129
```

The loss mask `m` is set to 0 at reference positions and 1 everywhere else — the model is not penalized for reconstructing what it already sees.

---

## Training Configuration

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| LR schedule | halved every 300k iterations |
| Batch size | 64 |
| Batch expansion K_e | 4 (one audio → 4 crops) |
| σ_min | 1e-8 |
| p_uncond | 0.05 |
| Reference crop length | 0.2–9 s, ≤50% of utterance |
| Total iterations | 700,000 |

### Running Stage 2

```bash
python src/train_text_to_latent.py --config configs/tts.json
```

---

## Key Config Knobs (`configs/tts.json`)

| Key | Default | Effect |
|---|---|---|
| `ttl.vector_field.proj_in.odim` | 512 | Hidden dim — scales VFE quadratically |
| `ttl.vector_field.main_blocks.n_blocks` | 4 | Number of superblocks |
| `ttl.vector_field.main_blocks.convnext_0.dilation_lst` | [1,2,4,8] | Receptive field per superblock |
| `ttl.text_encoder.convnext.num_layers` | 6 | Text ConvNeXt depth |
| `ttl.text_encoder.attn_encoder.n_layers` | 4 | Text self-attention depth |
| `ttl.chunk_compress_factor` | 6 | Temporal compression ratio |

### Reducing model size

A practical small TTL (~10M params): set `proj_in.odim=256`, `n_blocks=2`, `convnext.num_layers=4`.
