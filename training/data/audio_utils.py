import torch
import torchaudio

def ensure_sr(wav: torch.Tensor, sr_in: int, sr_out: int, device=None):
    """
    High-quality resampling function to ensure consistent audio quality.
    
    Args:
        wav: Input tensor [T] or [1, T] or [B, 1, T]
        sr_in: Input sample rate
        sr_out: Target sample rate
        device: Device to move the tensor to (optional)
        
    Returns:
        Resampled tensor on the specified device.
    """
    if device is None:
        device = wav.device
        
    # Ensure [..., T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        
    if sr_in != sr_out:
        wav = torchaudio.functional.resample(
            wav, sr_in, sr_out,
            lowpass_filter_width=64,  # higher = cleaner
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    return wav.to(device)



