import torch
import torch.nn as nn
from .modules import ConvNeXtBlock, LayerNorm1d

class LatentEncoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        
        # Defaults based on your JSON if cfg is missing
        if cfg is None:
            cfg = {
                "ksz": 7,
                "hdim": 512,
                "intermediate_dim": 2048,
                "dilation_lst": [1] * 10,
                "odim": 24,
                "idim": 1253
            }

        in_channels = cfg.get('idim', 1253)
        dim = cfg['hdim']
        out_channels = cfg['odim']
        kernel_size = cfg['ksz']
        intermediate_dim = cfg['intermediate_dim']
        dilations = cfg['dilation_lst']

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(dim),
        )

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim, 
                intermediate_dim=intermediate_dim, 
                kernel_size=kernel_size, 
                dilation=d
            )
            for d in dilations
        ])
        
        self.proj = nn.Conv1d(dim, out_channels, kernel_size=1)
        self.norm = LayerNorm1d(out_channels)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        x = self.norm(x)
        return x
