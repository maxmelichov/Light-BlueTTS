import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import LayerNorm as LayerNormWrapper

# -----------------------------------------------------------------------------
# Wrappers to match Checkpoint/Notebook Structure
# -----------------------------------------------------------------------------

class LinearWrapper(nn.Module):
    """
    Wraps nn.Linear to match checkpoint path `linear.weight` -> `module.linear.weight`.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ProjectionWrapper(nn.Module):
    """
    Wraps Conv1d 1x1 to match `proj.net.weight`.
    ONNX graph shows no bias for proj_in/proj_out Conv nodes.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------------------------------------------------------
# Basic Blocks
# -----------------------------------------------------------------------------

class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(embed_dim, scale=1000.0)
        # MLP: 64 -> 256 -> 64 (matches ONNX graph sources 160-164)
        self.mlp = nn.Sequential(
            LinearWrapper(embed_dim, embed_dim * 4),
            Mish(),
            LinearWrapper(embed_dim * 4, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(x)
        x = self.mlp(x)
        return x

class TimeCondBlock(nn.Module):
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        # Output channels for Shift (Add)
        self.linear = LinearWrapper(time_dim, channels)
        
        # Zero-initialize the projection so the block starts as Identity.
        nn.init.zeros_(self.linear.linear.weight)
        nn.init.zeros_(self.linear.linear.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        # time_emb: [B, time_dim]
        
        # Project time to Shift
        cond = self.linear(time_emb)       # [B, C]
        cond = cond.unsqueeze(-1)          # [B, C, 1]
        
        # Additive conditioning: x + shift
        return x + cond

class ConvNeXtBlock1D(nn.Module):
    """
    ConvNeXt Block with symmetric padding.
    NOTE: expansion=2 matches ONNX graph weights (512->1024).
    ONNX uses explicit Pad->Conv (no built-in Conv padding).
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        expansion: int = 2, 
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        # Symmetric padding: pad_left = pad_right = (kernel_size - 1) // 2 * dilation
        # Matches ONNX graph pattern: separate Pad node -> Conv(padding=0)
        self.pad = ((kernel_size - 1) // 2) * dilation
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=0, groups=dim, dilation=dilation
        )
        self.norm = LayerNormWrapper(dim)
        self.pwconv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * 1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None: x = x * mask
        residual = x
        
        # Symmetric replicate padding — verified against reference ONNX:
        # all 28 Pad nodes use mode='edge' (PyTorch 'replicate').
        # Pad values: [0, 0, pad, 0, 0, pad] (symmetric on time dim).
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.dwconv(x)
        if mask is not None: x = x * mask
        
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.dropout(x)
        
        x = x + residual
        if mask is not None: x = x * mask
        
        return x

class ConvNeXtStack(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        # Expansion=2 to match ONNX intermediate=1024 with hidden=512 (512*2=1024)
        # Note: If hidden=256, expansion=4 would be needed. But checkpoints use 512.
        self.convnext = nn.ModuleList([
            ConvNeXtBlock1D(channels, kernel_size=kernel_size, dilation=d, expansion=2)
            for d in dilations
        ])

    def forward(self, x, mask=None):
        for blk in self.convnext:
            x = blk(x, mask)
        return x

# -----------------------------------------------------------------------------
# Attention Mechanisms
# -----------------------------------------------------------------------------

def apply_rotary_pos_emb(x: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    """
    x:   [B, H, T, D] or [H, B, T, D]
    cos: [T, D/2] or [B, T, D/2] or [1, B, T, D/2]
    sin: [T, D/2] or [B, T, D/2] or [1, B, T, D/2]
    """
    # Note: B, H here are just placeholders for first two dims
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    x1 = x[..., : D // 2]  # [..., T, D/2]
    x2 = x[..., D // 2 :]  # [..., T, D/2]

    # Handle batched or unbatched freqs
    if cos.dim() == 2:
        cos = cos[None, None, :, :]  # [1, 1, T, D/2]
        sin = sin[None, None, :, :]  # [1, 1, T, D/2]
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)       # [B, 1, T, D/2]
        sin = sin.unsqueeze(1)       # [B, 1, T, D/2]

    # (x1, x2) rotated in the complex plane
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return torch.cat([x1_rot, x2_rot], dim=-1)

class AttentionModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_context: int,
        num_heads: int,
        attn_dim: int,
        use_rope: bool,
        dropout: float = 0.0,
        rope_gamma: float = 10.0,  # LARoPE scaling γ
        attn_scale: Optional[float] = None,  # Override attention scale divisor
    ):
        super().__init__()
        assert attn_dim % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.attn_dim = attn_dim
        self.use_rope = use_rope
        self.rope_gamma = rope_gamma
        # ONNX graph shows all attention blocks use sqrt(attn_dim) = sqrt(256) = 16.0
        # as the scale divisor, NOT sqrt(head_dim). Default to sqrt(attn_dim).
        self.attn_scale = attn_scale if attn_scale is not None else math.sqrt(self.attn_dim)
        
        # Wrapped Linears
        self.W_query = LinearWrapper(d_model, attn_dim)
        self.W_key = LinearWrapper(d_context, attn_dim)
        self.W_value = LinearWrapper(d_context, attn_dim)
        self.out_fc = LinearWrapper(attn_dim, d_model)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        if use_rope:
            # Frequencies for half the head dimension
            # ONNX: 'theta' shape [1, 1, 32] (head_dim/2=32), 'increments' shape [1, 1000, 1] int64.
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            # Pre-scale by rope_gamma to match trace (Div -> Mul(theta))
            theta = (inv_freq * rope_gamma).view(1, 1, -1)  # [1, 1, D/2] matches ONNX shape
            
            self.register_buffer("theta", theta, persistent=True)
            self.register_buffer("increments", torch.arange(1000).view(1, 1000, 1), persistent=True)
            self.tanh = None
        else:
            self.theta = None
            self.increments = None
            # Style attention path: tanh on keys
            self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_keys: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, d_model, T] — query source (latent)
            context: [B, L, d_context] — key/value source, always channel-last.
                     ONNX: text is transposed before reaching here,
                           style_ttl is already [B, 50, 256].
        """
        B, d_model, T = x.shape
        L = context.shape[1]
        
        x_t = x.transpose(1, 2)  # [B, T, d_model]
        
        q = self.W_query(x_t)
        
        # Determine Keys
        if context_keys is not None:
            if context_keys.dim() == 2:
                context_keys = context_keys.unsqueeze(0)
            if context_keys.shape[0] == 1:
                context_keys = context_keys.expand(B, -1, -1)
            k = self.W_key(context_keys)
        else:
            k = self.W_key(context)
            
        v = self.W_value(context)
        
        # Style attention (no RoPE): tanh on keys
        if not self.use_rope and self.tanh is not None:
            k = self.tanh(k)

        H = self.num_heads
        D = self.head_dim
        
        # ONNX layout: [H, B, T, D]
        q = q.view(B, T, H, D).permute(2, 0, 1, 3)  # [H, B, T, D]
        k = k.view(B, L, H, D).permute(2, 0, 1, 3)  # [H, B, L, D]
        v = v.view(B, L, H, D).permute(2, 0, 1, 3)  # [H, B, L, D]
        
        if self.use_rope:
            # ---------------- Length-Aware RoPE (LARoPE) ----------------
            # Ref: https://arxiv.org/abs/2509.11084
            # Normalizes positions by sequence length to induce diagonal bias.
            #
            # ONNX trace (1-to-1):
            #   ReduceSum(mask) -> Reshape -> [B,1,1]  (=sequence length)
            #   Slice(increments, :T) -> [1,T,1]  (=positions)
            #   Div([1,T,1], [B,1,1]) -> [B,T,1]  (=normalized positions)
            #   Mul([B,T,1], theta[1,1,D/2]) -> [B,T,D/2]  (=frequencies)
            #   Sin / Cos -> [B,T,D/2]
            device = x.device
            
            # Get lengths from masks: sum all non-batch dims -> [B] -> reshape [B,1,1]
            if x_mask is not None:
                len_q = x_mask.sum(dim=(-2, -1)).reshape(-1, 1, 1)  # [B,1,1]
            else:
                len_q = torch.tensor([T], device=device, dtype=torch.float32).reshape(1, 1, 1)

            if context_mask is not None:
                len_k = context_mask.sum(dim=(-2, -1)).reshape(-1, 1, 1)  # [B,1,1]
            else:
                len_k = torch.tensor([L], device=device, dtype=torch.float32).reshape(1, 1, 1)
            
            # Positions: Slice increments [1,1000,1] -> [1,T,1] / [1,L,1], cast to float
            if self.increments is not None and self.increments.shape[1] >= max(T, L):
                pos_q = self.increments[:, :T, :].to(device).float()  # [1, T, 1]
                pos_k = self.increments[:, :L, :].to(device).float()  # [1, L, 1]
            else:
                pos_q = torch.arange(T, device=device, dtype=torch.float32).reshape(1, -1, 1)
                pos_k = torch.arange(L, device=device, dtype=torch.float32).reshape(1, -1, 1)
            
            # Normalize: [1,T,1] / [B,1,1] -> [B,T,1]
            norm_pos_q = pos_q / len_q  # [B, T, 1]
            norm_pos_k = pos_k / len_k  # [B, L, 1]
            
            # Frequencies: [B,T,1] * [1,1,D/2] -> [B,T,D/2]
            theta = self.theta if self.theta is not None else (
                (1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))) * self.rope_gamma
            ).view(1, 1, -1)
            
            freqs_q = norm_pos_q * theta  # [B, T, D/2]
            freqs_k = norm_pos_k * theta  # [B, L, D/2]
            
            cos_q, sin_q = freqs_q.cos(), freqs_q.sin()  # [B, T, D/2]
            cos_k, sin_k = freqs_k.cos(), freqs_k.sin()  # [B, L, D/2]
            
            # Unsqueeze for [H, B, T, D] layout broadcasting -> [1, B, T, D/2]
            cos_q, sin_q = cos_q.unsqueeze(0), sin_q.unsqueeze(0)
            cos_k, sin_k = cos_k.unsqueeze(0), sin_k.unsqueeze(0)

            # Apply rotation separately to Q and K
            q = apply_rotary_pos_emb(q, cos_q, sin_q)  # [H, B, T, D]
            k = apply_rotary_pos_emb(k, cos_k, sin_k)  # [H, B, L, D]
            # ------------------------------------------------------------
        
        # Scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / self.attn_scale
        
        # Pre-softmax: mask invalid keys (context positions)
        # Matches ONNX: Equal(text_mask==0) -> Where(-inf, logits)
        if context_mask is not None:
            if context_mask.dim() == 2:
                context_mask = context_mask.unsqueeze(1)  # [B,1,L]
            cm = (context_mask == 0)  # [B, 1, L] bool
            # [H, B, T, L] vs [1, B, 1, L]
            attn_logits = attn_logits.masked_fill(cm.unsqueeze(0), float('-inf'))
            
        attn = torch.softmax(attn_logits, dim=-1)
        
        # Post-softmax: zero out attention for masked query positions
        # Matches ONNX: Equal_1(latent_mask==0) -> Where_1(0, softmax)
        if x_mask is not None:
            if x_mask.dim() == 2:
                x_mask = x_mask.unsqueeze(1)  # [B,1,T]
            # x_mask: [B, 1, T] -> [1, B, T, 1]
            qm = (x_mask == 0).permute(1, 0, 2).unsqueeze(-1) # [1, B, T, 1]
            attn = attn.masked_fill(qm, 0.0)
            
        out = torch.matmul(attn, v)  # [H, B, T, D]
        
        # [H, B, T, D] -> [B, T, H, D] -> [B, T, H*D]
        out = out.permute(1, 2, 0, 3).contiguous().view(B, T, self.attn_dim)
        out = self.out_fc(out)
        out = self.dropout(out)
        
        # Mask output (matches ONNX Mul_14: out_fc output * latent_mask_transposed)
        if x_mask is not None:
            out = out * x_mask.transpose(1, 2)  # x_mask [B,1,T] -> [B,T,1]
        
        out = out.transpose(1, 2)  # [B, C, T]
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_context: int,
        num_heads: int = 8,
        attn_dim: int = 256,
        use_rope: bool = True,
        rope_gamma: float = 10.0,
        attn_scale: Optional[float] = None,
    ):
        super().__init__()
        self.use_rope = use_rope
        attn_module = AttentionModule(
            d_model, d_context, num_heads, attn_dim, use_rope,
            rope_gamma=rope_gamma, attn_scale=attn_scale,
        )
        # Match checkpoint naming conventions:
        # Text (RoPE) -> 'attn'
        # Style (No RoPE) -> 'attention'
        if use_rope:
            self.attn = attn_module
        else:
            self.attention = attn_module
            
        self.norm = LayerNormWrapper(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_keys: Optional[torch.Tensor],
        x_mask: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, d_model, T]
            context: [B, L, d_context] — always channel-last.
                     Caller is responsible for transposing if needed.
        """
        # Pre-mask x before attention (matches ONNX: main_blocks.N/Mul before Transpose)
        if x_mask is not None:
            x = x * x_mask
        
        residual = x
        
        # Attention
        if self.use_rope:
            attn_out = self.attn(x, context, context_keys, x_mask, context_mask)
        else:
            attn_out = self.attention(x, context, context_keys, x_mask, context_mask)
        
        # Residual Add
        x = residual + attn_out
        
        # Norm
        x = self.norm(x)
        
        if x_mask is not None:
            x = x * x_mask
            
        return x

# -----------------------------------------------------------------------------
# Main Estimator
# -----------------------------------------------------------------------------

class VectorFieldEstimator(nn.Module):
    """
    Vector Field Estimator for Flow Matching TTS.
    
    Architecture verified against ONNX trace in `checks/notebook.ipynb`.
    """
    def __init__(
        self,
        in_channels: int = 144,
        hidden_channels: int = 512,  # Checkpoint: 512
        out_channels: int = 144,
        text_dim: int = 256,         # Checkpoint: 256
        style_dim: int = 256,        # Checkpoint: 256
        num_style_tokens: int = 50,  # tts.json: style_token_layer.n_style=50
        num_superblocks: int = 4,
        time_embed_dim: int = 64,    # Matches ONNX graph
        rope_gamma: float = 10.0,  # LARoPE scaling factor (tts.json: rotary_scale=10)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.style_dim = style_dim
        self.rope_gamma = rope_gamma

        # ------------------------------------------------------------------
        # Style keys (ONNX: baked-in constant expanded to batch via Tile)
        # ------------------------------------------------------------------
        # The ONNX VF graph shown in `checks/notebook.ipynb` does NOT take
        # `style_keys` as an input; instead it uses a constant learnable key
        # tensor of shape [1, 50, 256] that is expanded to [B, 50, 256].
        #
        # Training code in this repo may still pass `style_keys` explicitly.
        # We support both:
        # - if `style_keys` is provided -> use it
        # - else -> use this baked-in `style_key`
        self.style_key = nn.Parameter(torch.randn(1, num_style_tokens, style_dim) * 0.02)
        
        # 1. Input Projection
        self.proj_in = ProjectionWrapper(in_channels, hidden_channels)
        
        # 2. Time Encoder
        self.time_encoder = TimeEncoder(time_embed_dim)
        
        # 3. Main Blocks
        self.main_blocks = nn.ModuleList()
        
        # ONNX graph shows all attention blocks share the same scale divisor
        # sqrt(attn_dim) = sqrt(256) = 16.0 for both text and style attention.
        # This is verified from the original ONNX constants (Constant_39 and
        # Constant_9 in each attention block all equal 16.0).
        shared_attn_scale = math.sqrt(256)  # = sqrt(attn_dim) = 16.0
        
        for _ in range(num_superblocks):
            # 0: 4x Dilated ConvNeXt
            self.main_blocks.append(
                ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1, 2, 4, 8])
            )
            # 1: Time Injection
            self.main_blocks.append(
                TimeCondBlock(time_dim=time_embed_dim, channels=hidden_channels)
            )
            # 2: 1x Standard ConvNeXt
            self.main_blocks.append(
                ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1])
            )
            # 3: Text Attention (Length-Aware RoPE / LARoPE)
            self.main_blocks.append(
                CrossAttentionBlock(
                    d_model=hidden_channels,
                    d_context=text_dim,
                    num_heads=4,
                    attn_dim=256,
                    use_rope=True,
                    rope_gamma=self.rope_gamma,
                    attn_scale=shared_attn_scale,
                )
            )
            # 4: 1x Standard ConvNeXt
            self.main_blocks.append(
                ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1])
            )
            # 5: Style Attention (No RoPE, Tanh on Keys)
            self.main_blocks.append(
                CrossAttentionBlock(
                    d_model=hidden_channels,
                    d_context=style_dim,
                    num_heads=2,
                    attn_dim=256,
                    use_rope=False,
                    attn_scale=shared_attn_scale,
                )
            )
            
        # 4. Last ConvNeXt
        self.last_convnext = ConvNeXtStack(
            hidden_channels, kernel_size=5, dilations=[1, 1, 1, 1]
        )
        
        # 5. Output Projection
        self.proj_out = ProjectionWrapper(hidden_channels, out_channels)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        style_ttl: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        current_step: Optional[torch.Tensor] = None,
        total_step: Optional[torch.Tensor] = None,
        style_keys: Optional[torch.Tensor] = None,
        # Training aliases (accepted via kwargs)
        text_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass supporting both ONNX inference and training calling conventions.

        ONNX inference call:
            model(noisy_latent, text_emb, style_ttl, latent_mask, text_mask,
                  current_step, total_step, style_keys=...)
            -> returns denoised_latent (Euler step applied)

        Training call:
            model(noisy_latent=x_t, text_context=h, style_ttl=ref_values,
                  style_keys=ref_keys, latent_mask=mask, text_mask=tmask,
                  current_step=t)
            -> returns raw vector field prediction (no Euler step)

        Args:
            noisy_latent:  [B, 144, latent_length]
            text_emb:      [B, 256, text_length]  (ONNX name)
            text_context:  [B, 256, text_length]  (training alias for text_emb)
            style_ttl:     [B, 50, 256]
            latent_mask:   [B, 1, latent_length]
            text_mask:     [B, 1, text_length]
            current_step:  [B] — normalized time t in [0,1] (training), or discrete step (ONNX)
            total_step:    [B] — total steps (ONNX only; None during training)
            style_keys:    [B, 50, 256] optional; if None, uses internal `style_key`

        Returns:
            Training (total_step is None):
                vector_field: [B, 144, latent_length]
            ONNX inference (total_step provided):
                denoised_latent: [B, 144, latent_length]
        """
        # Resolve text_emb from either name
        if text_emb is None:
            text_emb = text_context
        assert text_emb is not None, "Must provide text_emb or text_context"

        B = noisy_latent.shape[0]

        # ONNX graph uses a baked-in style key expanded to batch.
        # Keep support for explicit `style_keys` (training / legacy callers).
        if style_keys is None:
            style_keys = self.style_key.expand(B, -1, -1)

        # --- Time normalization ---
        if total_step is not None:
            # ONNX inference path: /Reshape, /Reshape_1, /Div
            t_norm = current_step.reshape(B, 1, 1) / total_step.reshape(B, 1, 1)  # [B,1,1]
            reciprocal = (1.0 / total_step.reshape(B, 1, 1))  # [B,1,1]
            t_norm_flat = t_norm.reshape(B)  # [B]
        else:
            # Training path: current_step is already normalized t in [0, 1]
            t_norm_flat = current_step.reshape(B)  # [B]

        # /vector_field/time_encoder — sinusoidal → MLP → [B, 64]
        t_emb = self.time_encoder(t_norm_flat)  # [B, 64]

        # /vector_field/main_blocks.3/Transpose_1 — text [B,256,T] → [B,T,256]
        text_blc = text_emb.transpose(1, 2)  # [B, T, 256]

        # /vector_field/proj_in/net/Conv + /vector_field/proj_in/Mul
        x = self.proj_in(noisy_latent)
        if latent_mask is not None:
            x = x * latent_mask

        # Main Blocks
        for i, block in enumerate(self.main_blocks):
            idx_in_super = i % 6

            if idx_in_super == 0:
                x = block(x, mask=latent_mask)
            elif idx_in_super == 1:
                x = block(x, t_emb)
                if latent_mask is not None:
                    x = x * latent_mask
            elif idx_in_super == 2:
                x = block(x, mask=latent_mask)
            elif idx_in_super == 3:
                x = block(x, context=text_blc, context_keys=None,
                          x_mask=latent_mask, context_mask=text_mask)
            elif idx_in_super == 4:
                x = block(x, mask=latent_mask)
            elif idx_in_super == 5:
                x = block(x, context=style_ttl, context_keys=style_keys,
                          x_mask=latent_mask, context_mask=None)

        # /vector_field/last_convnext
        x = self.last_convnext(x, mask=latent_mask)

        # /vector_field/proj_out/net/Conv + /vector_field/proj_out/Mul
        diff_out = self.proj_out(x)
        if latent_mask is not None:
            diff_out = diff_out * latent_mask

        if total_step is not None:
            # ONNX inference: /Mul (reciprocal * diff_out) → /Add (noisy + ...) → /Mul_1 (* mask)
            denoised = noisy_latent + reciprocal * diff_out
            return denoised * latent_mask if latent_mask is not None else denoised
        else:
            # Training: return raw vector field prediction
            return diff_out