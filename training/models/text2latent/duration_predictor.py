import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.text2latent.text_encoder import (
    AttnEncoder,
    TextEmbedderWrapper,
    ConvNeXtWrapper,
    ConvNeXtBlock,
    LayerNorm
)

class DPReferenceEncoder(nn.Module):
    """
    A.3.1 DP Reference Encoder
    - Input: Compressed Latents (144)
    - Output: 64-dim reference embedding (Paper: "stacking... resulting in a 64-dimensional vector")
    """
    def __init__(self, in_channels=144, d_model=64, hidden_dim=256, num_blocks=4, num_queries=8, query_dim=16):
        super().__init__()
        
        # Linear 144 -> 64
        self.input_proj = nn.Conv1d(in_channels, d_model, 1)
        
        # 4 ConvNeXt blocks (dim 256 -> mlp_ratio=4 if d_model=64)
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(d_model, expansion_factor=hidden_dim // d_model)
            for _ in range(num_blocks)
        ])
        
        # 2 Cross-Attention Layers
        self.num_queries = num_queries
        self.query_dim = query_dim  # e.g. 16 -> 8 * 16 = 128
        
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, self.query_dim) * 0.02)
        
        # Cross Attention layers
        # Input (Context) is d_model=64.
        # We need to project Context to Key/Value.
        # Queries are Fixed.
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.query_dim, num_heads=1, kdim=d_model, vdim=d_model, batch_first=True)
            for _ in range(2)
        ])
        
    def forward(self, z_ref, mask=None):
        """
        z_ref: [B, 144, T]
        mask: [B, 1, T]
        """
        B = z_ref.shape[0]
        
        # 1. Linear + ConvNeXt
        x = self.input_proj(z_ref) # [B, 64, T]
        
        if mask is not None:
            x = x * mask
            
        for blk in self.convnext:
            x = blk(x, mask=mask)
            
        # x: [B, 64, T] -> [B, T, 64] (Context for Attn)
        context = x.transpose(1, 2)
        
        # Prepare Queries: [B, 8, 8]
        q = self.queries.expand(B, -1, -1)
        
        # Mask handling
        key_padding_mask = None
        if mask is not None:
            # mask is 1 for valid. key_padding_mask needs True for Pad.
            key_padding_mask = (mask.squeeze(1) == 0) # [B, T]
            
        # 2. Cross Attention
        for layer in self.attn_layers:
            out, _ = layer(q, context, context, key_padding_mask=key_padding_mask)
            q = q + out # Residual
            
        # 3. Stack
        # q: [B, 8, 16] -> [B, 128]
        out = q.reshape(B, -1)
        return out


class DPTextEncoder(nn.Module):
    """
    A.3.2 DP Text Encoder
    - Input: Text IDs
    - Output: 64-dim utterance-level text embedding
    
    Note: The pre-trained ONNX uses vocab_size=163 (char_embedder weight [163, 64]).
    When training from scratch with a reduced alphabet, vocab_size=37 is used.
    This is the ONLY weight shape that differs from the ONNX; all other
    parameters (98 total in the style path) match 1-to-1.
    """
    def __init__(self, vocab_size=37, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        self.text_embedder = TextEmbedderWrapper(vocab_size, d_model)
        
        # 6 ConvNeXt blocks (intermediate 256 -> mlp_ratio=4)
        self.convnext = ConvNeXtWrapper(d_model, n_layers=6, expansion_factor=4)
        
        # Utterance token (prepend)
        self.sentence_token = nn.Parameter(torch.randn(1, d_model, 1) * 0.02)
        
        # 2 Self-Attention Blocks (256 filter, 2 heads, RoPE)
        self.attn_encoder = AttnEncoder(
            channels=d_model,
            n_heads=2,
            filter_channels=d_model * 4, # 256
            n_layers=2
        )
        
        # Final Projection (Conv1d 1x1, no bias - matches ONNX proj_out/net/Conv)
        self.proj_out = nn.Sequential()
        self.proj_out.add_module("net", nn.Conv1d(d_model, d_model, 1, bias=False))

    def forward(self, text_ids, mask=None):
        B, T = text_ids.shape
        
        # Embed
        x = self.text_embedder(text_ids) # [B, T, 64]
        
        x = x.transpose(1, 2) # [B, 64, T]
        
        if mask is not None:
            x = x * mask

        # Prepend Utterance Token - FIXED LOCATION (Before ConvNeXt)
        u_token = self.sentence_token.expand(B, -1, -1) # [B, 64, 1]
        x = torch.cat([u_token, x], dim=2) # [B, 64, T+1]
        
        # Update mask
        if mask is not None:
            # Add 1 for utterance token (valid)
            mask_u = torch.ones(B, 1, 1, device=mask.device)
            mask = torch.cat([mask_u, mask], dim=2)
            
        # ConvNeXt
        x = self.convnext(x, mask=mask)
            
        # Store for residual
        conv_out = x

        # Attention
        x = self.attn_encoder(x, mask=mask)
        
        # Residual (ConvNeXt output + Attention output)
        x = x + conv_out
            
        # Take first token (utterance token)
        # Slice: [B, 64, 1]
        first_token = x[:, :, :1] 
        
        # Linear/Conv
        out = self.proj_out(first_token) # [B, 64, 1]
        
        if mask is not None:
            out = out * mask[:, :, :1]
        
        return out.squeeze(2) # [B, 64]


class DurationEstimator(nn.Module):
    """
    A.3.3 Duration Estimator
    - Input: Text Emb (64) + Style Emb (64 or 128)
    - Output: Scalar Duration
    """
    def __init__(self, text_dim=64, style_dim=128):
        super().__init__()
        # Input is 64 (text) + 128 (style) = 192
        
        # Structure matched to logs: layers.0 -> activation -> layers.1
        self.layers = nn.ModuleList([
            nn.Linear(text_dim + style_dim, 128), # Input 192, Hidden 128
            nn.Linear(128, 1)
        ])
        self.activation = nn.PReLU()

    def forward(self, text_emb, style_emb, text_mask=None, return_log=False):
        # text_emb: [B, 64]
        # style_emb: [B, 64] or [B, N, D]
        if style_emb.dim() > 2:
            style_emb = style_emb.reshape(style_emb.shape[0], -1)
            
        x = torch.cat([text_emb, style_emb], dim=1) # [B, 192]
        
        x = self.layers[0](x)
        x = self.activation(x)
        x = self.layers[1](x) # [B, 1]
        
        if return_log:
            return x.squeeze(1)
            
        return torch.exp(x).squeeze(1)


class TTSDurationModel(nn.Module):
    def __init__(self, vocab_size=37, style_tokens=8, style_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.sentence_encoder = DPTextEncoder(vocab_size=vocab_size)
        self.ref_encoder = DPReferenceEncoder(num_queries=style_tokens, query_dim=style_dim)
        self.predictor = DurationEstimator(text_dim=64, style_dim=style_tokens * style_dim)

    def forward(self, text_ids, z_ref=None, text_mask=None, ref_mask=None, style_tokens=None, return_log=False):
        """
        Args:
            text_ids: [B, T]
            z_ref: [B, 144, T_ref] (optional if style_tokens provided)
            text_mask: [B, 1, T]
            ref_mask: [B, 1, T_ref]
            style_tokens: [B, 8, 16] (optional pre-computed style tokens)
            return_log: If True, return log(duration). Else return duration (linear).
            
        Returns:
            duration: [B] (scalar)
        """
        text_emb = self.sentence_encoder(text_ids, mask=text_mask) # [B, 64]
        
        if style_tokens is not None:
            style_emb = style_tokens
        elif z_ref is not None:
            style_emb = self.ref_encoder(z_ref, mask=ref_mask)         # [B, 128]
        else:
            raise ValueError("Either z_ref or style_tokens must be provided")
        
        # Original paper: No explicit length feature.
        # The utterance token embedding should contain all necessary duration info via attention.
        duration = self.predictor(text_emb, style_emb, text_mask=text_mask, return_log=return_log) # [B]
        
        return duration
