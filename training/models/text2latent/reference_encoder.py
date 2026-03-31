import math
import torch
import torch.nn as nn

from models.text2latent.text_encoder import ConvNeXtWrapper

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] sequence
        Returns:
            pe: [1, T, C] positional encoding
        """
        T = x.shape[1]
        # In ONNX, slicing pe with a dynamic T can be tricky if T is not inferred as dynamic.
        # But legacy trace should record the Slice op.
        return self.pe[:, :T, :]

class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 144,
        d_model: int = 256,      # User Requirement
        hidden_dim: int = 1024,  # User Requirement
        num_blocks: int = 6,
        num_tokens: int = 50,
        num_heads: int = 4,      # tts.json: style_token_layer.n_heads=2
    ):
        super().__init__()
        self.d_model = d_model

        # --- Ratio Calculation ---
        # 1024 // 256 = 4. 
        # The ConvNeXt block will expand 256 -> 1024 -> 256 internally.
        if hidden_dim % d_model != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by d_model ({d_model})")
        
        mlp_ratio = hidden_dim // d_model 

        # 1. Linear Projection (Maps 144 -> 256)
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        
        # 2. ConvNeXt Blocks
        # Uses ConvNeXtWrapper for consistent weight naming (norm.norm.weight)
        # with the rest of the codebase (text_encoder, duration_predictor).
        # Keys: convnext.convnext.N.{dwconv,norm,pwconv1,pwconv2,gamma}
        self.convnext = ConvNeXtWrapper(d_model, n_layers=num_blocks, expansion_factor=mlp_ratio)
        
        self.pos_emb = SinusoidalPositionalEmbedding(d_model)
        
        # 3. Learnable Vectors (Ref Keys)
        # These act as Queries in this module to extract info from the audio
        self.ref_keys = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)

        # 4. Cross-Attention Layers
        self.attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
            })
            for _ in range(2)
        ])

    def forward(self, z_ref: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            z_ref: [B, 144, T] Compressed Latents
            mask:  [B, 1, T] Binary mask (1=valid, 0=padding)
        Returns:
            ref_values: [B, num_tokens, d_model] (Context-aware embedding)
            ref_keys:   [B, num_tokens, d_model] (Static learnable tokens)
        """
        B, C, T = z_ref.shape
        
        # --- 1. ConvNet Encoding ---
        # Input Projection: [B, 144, T] -> [B, 256, T]
        x = self.input_proj(z_ref)
        
        # ConvNeXt Blocks
        x = self.convnext(x, mask=mask)
            
        # Prepare Key/Value Sequence for Attention
        # Transpose to [B, T, 256] for LayerNorm and Attention
        kv_seq = x.transpose(1, 2) 
        
        # Add Positional Embeddings to the Audio features
        kv_seq = kv_seq + self.pos_emb(kv_seq)

        # --- 2. Mask Preparation ---
        # Input mask is [B, 1, T] (1=keep, 0=drop).
        # MultiheadAttention `key_padding_mask` expects (B, T) where True=drop.
        key_padding_mask = None
        if mask is not None:
            # Squeeze to [B, T]. Logic: If mask value is 0, make it True (ignore).
            key_padding_mask = (mask.squeeze(1) == 0)

        # --- 3. Cross-Attention Loop ---
        # Query: The learnable tokens, expanded to batch size [B, 50, 256]
        q = self.ref_keys.unsqueeze(0).expand(B, -1, -1) 

        for layer in self.attn_layers:
            # Pre-Norm Architecture
            q_norm = layer["norm_q"](q)
            kv_norm = layer["norm_kv"](kv_seq)
            
            # Attention: Q = Learnable Tokens, K/V = Encoded Audio
            # We extract information FROM audio (kv) INTO the tokens (q)
            attn_out, _ = layer["attn"](
                query=q_norm,
                key=kv_norm,
                value=kv_norm,
                key_padding_mask=key_padding_mask
            )
            
            # Residual Connection + FFN
            q = q + attn_out
            q = q + layer["ffn"](q)

        # Returns:
        # 1. Ref Values: The output of the attention (contextualized audio summary)
        # 2. Ref Keys: The original learnable tokens (static, used for matching in TextEncoder)
        return q, self.ref_keys.unsqueeze(0).expand(B, -1, -1)