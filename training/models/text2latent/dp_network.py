import torch
import torch.nn as nn

from .duration_predictor import TTSDurationModel


class DPNetwork(TTSDurationModel):
    """
    Backward-compatible wrapper around TTSDurationModel.

    Uses inheritance (not composition) so that state_dict keys
    are ``sentence_encoder.*`` / ``predictor.*`` — matching the
    ONNX graph and existing checkpoints (no ``core.`` prefix).
    """

    def __init__(
        self,
        vocab_size: int = 37,
        latent_channels: int = 144,   # kept for call-site compat, unused
        style_tokens: int = 8,
        style_dim: int = 16,
    ):
        super().__init__(
            vocab_size=vocab_size,
            style_tokens=style_tokens,
            style_dim=style_dim,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        z_ref: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        style_tokens: torch.Tensor | None = None,
        return_log: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            text_ids:     [B, T_text]
            z_ref:        [B, C_latent, T_ref]  (normalized compressed latents)
            text_mask:    [B, 1, T_text]         (1 = valid, 0 = pad)
            ref_mask:     [B, 1, T_ref] or None  (1 = valid, 0 = pad)
            style_tokens: [B, N, D] or None      (pre-computed style tokens)
            return_log:   If True return log(duration), else linear duration.

        Returns:
            duration: [B]
        """
        # Ensure masks are float32
        if text_mask is not None and text_mask.dtype != torch.float32:
            text_mask = text_mask.float()

        if ref_mask is not None and ref_mask.dtype != torch.float32:
            ref_mask = ref_mask.float()
        elif ref_mask is None and z_ref is not None:
            B, C, T_ref = z_ref.shape
            ref_mask = torch.ones(B, 1, T_ref, device=z_ref.device, dtype=torch.float32)

        return super().forward(
            text_ids,
            z_ref=z_ref,
            text_mask=text_mask,
            ref_mask=ref_mask,
            style_tokens=style_tokens,
            return_log=return_log,
        )
