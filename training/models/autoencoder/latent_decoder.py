from collections import OrderedDict

import torch
import torch.nn as nn

from .modules import CausalConv1d, CausalConvNeXtBlock


class CausalInputProjection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.net = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.net(x)


class FinalBatchNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(x)


class VocoderHead(nn.Module):
    def __init__(self, dim=512, hdim=2048, out_dim=512, kernel_size=3):
        super().__init__()
        # Match the AE decoder head config:
        # layer1 causal conv -> PReLU -> layer2 1x1 conv -> transpose -> reshape.
        self.layer1 = CausalInputProjection(dim, hdim, kernel_size=kernel_size)
        self.act = nn.PReLU()
        self.layer2 = nn.Conv1d(hdim, out_dim, kernel_size=1, bias=False)

        # Keep the final projection small at initialization.
        self.layer2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)  # [B, 512, T]

        # Match notebook trace:
        # [B, 512, T] -> [B, T, 512] -> [B, T * 512]
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(x.shape[0], -1)
        return x

class LatentDecoder1D(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        # Defaults based on configs/tts.json -> ae.decoder.
        if cfg is None:
            cfg = {
                "idim": 24,
                "hdim": 512,
                "intermediate_dim": 2048,
                "ksz": 7,
                "dilation_lst": [1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
                "head": {"idim": 512, "hdim": 2048, "odim": 512, "ksz": 3},
                "chunk_compress_factor": 1,
                "normalizer_scale": 1.0,
            }

        in_channels = cfg['idim']
        dim = cfg['hdim']
        intermediate_dim = cfg['intermediate_dim']
        kernel_size = cfg['ksz']
        dilations = cfg['dilation_lst']

        self.input_channels = in_channels
        self.chunk_compress_factor = int(cfg.get('chunk_compress_factor', 1))
        self.normalizer_scale = float(cfg.get('normalizer_scale', 1.0))
        self.compressed_channels = int(
            cfg.get('compressed_idim', in_channels * self.chunk_compress_factor)
        )

        # Match traced decoder buffers.
        self.register_buffer('latent_mean', torch.zeros(1, in_channels, 1), persistent=False)
        self.register_buffer('latent_std', torch.ones(1, in_channels, 1), persistent=False)

        self.embed = CausalInputProjection(in_channels, dim, kernel_size=kernel_size)

        self.convnext = nn.ModuleList([
            CausalConvNeXtBlock(
                dim=dim, 
                intermediate_dim=intermediate_dim, 
                kernel_size=kernel_size, 
                dilation=d
            )
            for d in dilations
        ])

        self.final_norm = FinalBatchNorm1d(dim)

        head_cfg = cfg['head']
        head_input_dim = int(head_cfg.get('idim', dim))
        if head_input_dim != dim:
            raise ValueError(
                f"Decoder head idim={head_input_dim} must match decoder hdim={dim}"
            )

        self.head = VocoderHead(
            dim=head_input_dim,
            hdim=head_cfg['hdim'],
            out_dim=head_cfg['odim'],
            kernel_size=int(head_cfg.get('ksz', 3)),
        )

    def _prepare_latents(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected latents with shape [B, C, T], got {tuple(x.shape)}")

        if x.shape[1] == self.input_channels:
            return x

        if x.shape[1] != self.compressed_channels:
            raise ValueError(
                f"Expected latent channels to be {self.input_channels} or "
                f"{self.compressed_channels}, got {x.shape[1]}"
            )

        if self.normalizer_scale != 0.0 and self.normalizer_scale != 1.0:
            x = x / self.normalizer_scale

        bsz, _, latent_len = x.shape
        x = x.reshape(bsz, self.input_channels, self.chunk_compress_factor, latent_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(bsz, self.input_channels, latent_len * self.chunk_compress_factor)
        return x

    def load_state_dict(self, state_dict, strict=True):
        import re

        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = key

            # --- embed (was input_conv) ---
            if key.startswith("input_conv.0."):
                new_key = key.replace("input_conv.0.", "embed.net.", 1)
            elif key.startswith("input_conv.1."):
                continue
            elif key.startswith("input_conv.net."):
                new_key = key.replace("input_conv.net.", "embed.net.", 1)

            # --- convnext (was blocks) ---
            elif key.startswith("blocks."):
                new_key = key.replace("blocks.", "convnext.", 1)
                m = re.match(r"convnext\.(\d+)\.dwconv\.(weight|bias)$", new_key)
                if m:
                    new_key = f"convnext.{m.group(1)}.dwconv.net.{m.group(2)}"

            # --- final_norm ---
            elif key.startswith("final_norm.") and not key.startswith("final_norm.norm."):
                new_key = key.replace("final_norm.", "final_norm.norm.", 1)

            # --- head ---
            elif key.startswith("head.conv1."):
                new_key = key.replace("head.conv1.", "head.layer1.net.", 1)
            elif key.startswith("head.conv2."):
                new_key = key.replace("head.conv2.", "head.layer2.", 1)

            if new_key == "head.layer2.bias":
                continue

            remapped[new_key] = value

        return super().load_state_dict(remapped, strict=strict)

    def forward(self, x):
        x = self._prepare_latents(x)
        x = x * self.latent_std + self.latent_mean

        x = self.embed(x)
        for block in self.convnext:
            x = block(x)

        x = self.final_norm(x)
        waveform = self.head(x)
        return waveform
