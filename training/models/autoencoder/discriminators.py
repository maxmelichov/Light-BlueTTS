import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        # Checklist channels: 1 -> 16 -> 64 -> 256 -> 512 -> 512 -> 1
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 16, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(16, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(64, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(512, 512, (kernel_size, 1), (1, 1), padding=(kernel_size//2, 0))), # Stride 1
        ])
        self.conv_post = norm_f(nn.Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # x: [B, 1, T]
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        if periods is None: periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorR(nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.n_fft, self.hop_length, self.win_length = resolution
        
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        
        # Architecture based on Table 7/Checklist
        # 1 -> 16 (5,5) s(1,1)
        # 16 -> 16 (5,5) s(2,1)
        # 16 -> 16 (5,5) s(2,1)
        # 16 -> 16 (5,5) s(2,1)
        # 16 -> 16 (5,5) s(1,1)
        # 16 -> 1 (3,3) s(1,1)
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 16, (5, 5), stride=(1, 1), padding=(2, 2))),
            norm_f(nn.Conv2d(16, 16, (5, 5), stride=(2, 1), padding=(2, 2))),
            norm_f(nn.Conv2d(16, 16, (5, 5), stride=(2, 1), padding=(2, 2))),
            norm_f(nn.Conv2d(16, 16, (5, 5), stride=(2, 1), padding=(2, 2))),
            norm_f(nn.Conv2d(16, 16, (5, 5), stride=(1, 1), padding=(2, 2))),
        ])
        self.conv_post = norm_f(nn.Conv2d(16, 1, (3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        
        # x: [B, 1, T]
        # Compute Linear Spectrogram
        # n_fft, hop, win from init
        # Window = Hann
        # spec = log( |STFT| + eps )
        
        if x.dim() == 3:
            x = x.squeeze(1) # [B, T]
            
        # Pad if necessary to avoid centering issues or ensure coverage? 
        # Usually torch.stft centers by default. Checklist doesn't specify padding details for STFT, 
        # but says "Take magnitude and log".
        
        x_stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length, 
                       window=self.window,
                       return_complex=True, center=True)
        x_mag = torch.abs(x_stft)
        x_log = torch.log(x_mag + 1e-9) # eps
        
        x = x_log.unsqueeze(1) # [B, 1, F, T_spec]
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions=None):
        super(MultiResolutionDiscriminator, self).__init__()
        if resolutions is None:
            # (n_fft, hop_length, win_length)
            # n_fft = [512, 1024, 2048]
            # hop = n_fft / 4 -> [128, 256, 512]
            # win = n_fft -> [512, 1024, 2048]
            resolutions = [
                (512, 128, 512),
                (1024, 256, 1024),
                (2048, 512, 2048)
            ]
        self.discriminators = nn.ModuleList([DiscriminatorR(r, use_spectral_norm=True) for r in resolutions])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
