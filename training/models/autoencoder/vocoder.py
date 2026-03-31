import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LayerNorm1d, ConvNeXtBlock

class VocoderHead(nn.Module):
    """
    Final projection and upsampling to audio.
    """
    def __init__(self, dim=512, intermediate_dim=2048):
        super().__init__()
        # Trace: decoder/head/layer1/net/Conv (Weight: [2048, 512, 3])
        self.layer1 = nn.Conv1d(dim, intermediate_dim, kernel_size=3, padding=1)
        
        # Trace: decoder/head/act/PRelu
        self.act = nn.PReLU(num_parameters=1) 
        
        # Trace: decoder/head/layer2/Conv (Weight: [512, 2048, 1])
        self.layer2 = nn.Conv1d(intermediate_dim, dim, kernel_size=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        
        # Trace ends with a Reshape to 'wav_tts'.
        # Since layer2 outputs 512 channels, and the target is audio,
        # this implies a Periodic Shuffle / PixelShuffle1D where the 
        # channels are folded into the time dimension.
        # [B, 512, T] -> [B, 1, T * 512]
        b, c, t = x.shape
        x = x.view(b, 1, c, t)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, 1, -1)
        
        return x

class StyleTTS2Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Mean/Std constants derived from trace names
        # tts.ae.latent_std: [1, 24, 1]
        self.register_buffer('latent_mean', torch.zeros(1, 24, 1))
        self.register_buffer('latent_std', torch.ones(1, 24, 1))

        # Initial Projection
        # Trace: decoder/embed/net/Conv (Weight: [512, 24, 7])
        self.input_conv = nn.Conv1d(24, 512, kernel_size=7, padding=3)
        
        # Backbone: 10 ConvNeXt Blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim=512, intermediate_dim=2048)
            for _ in range(10)
        ])
        
        # Final Norm
        # Trace: decoder/final_norm/BatchNormalization
        self.final_norm = nn.BatchNorm1d(512)
        
        # Head
        self.head = VocoderHead(dim=512)

    def load_from_checkpoint(self, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
            
        new_sd = {}
        
        # Mappings
        # Embed
        if 'decoder/embed/net/Conv.weight' in sd:
            new_sd['input_conv.weight'] = sd['decoder/embed/net/Conv.weight']
            new_sd['input_conv.bias'] = sd['decoder/embed/net/Conv.bias']
            
        # Final Norm
        if 'decoder/final_norm/BatchNormalization.weight' in sd:
            new_sd['final_norm.weight'] = sd['decoder/final_norm/BatchNormalization.weight']
            new_sd['final_norm.bias'] = sd['decoder/final_norm/BatchNormalization.bias']
            new_sd['final_norm.running_mean'] = sd['decoder/final_norm/BatchNormalization.running_mean']
            new_sd['final_norm.running_var'] = sd['decoder/final_norm/BatchNormalization.running_var']
            new_sd['final_norm.num_batches_tracked'] = sd['decoder/final_norm/BatchNormalization.num_batches_tracked']
            
        # Head
        if 'decoder/head/layer1/net/Conv.weight' in sd:
            new_sd['head.layer1.weight'] = sd['decoder/head/layer1/net/Conv.weight']
            new_sd['head.layer1.bias'] = sd['decoder/head/layer1/net/Conv.bias']
        
        if 'decoder/head/layer2/Conv.weight' in sd:
            new_sd['head.layer2.weight'] = sd['decoder/head/layer2/Conv.weight']
            # Check bias
            if 'decoder/head/layer2/Conv.bias' in sd:
                 new_sd['head.layer2.bias'] = sd['decoder/head/layer2/Conv.bias']
            
        # Blocks
        for i in range(10):
            prefix = f'decoder/convnext/{i}'
            target = f'blocks.{i}'
            
            # dwconv
            if f'{prefix}/dwconv/net/Conv.weight' in sd:
                new_sd[f'{target}.dwconv.weight'] = sd[f'{prefix}/dwconv/net/Conv.weight']
                new_sd[f'{target}.dwconv.bias'] = sd[f'{prefix}/dwconv/net/Conv.bias']
                
            # norm
            if f'{prefix}/norm/norm/LayerNormalization.weight' in sd:
                new_sd[f'{target}.norm.norm.weight'] = sd[f'{prefix}/norm/norm/LayerNormalization.weight']
                new_sd[f'{target}.norm.norm.bias'] = sd[f'{prefix}/norm/norm/LayerNormalization.bias']
                
            # pwconv1
            if f'{prefix}/pwconv1/Conv.weight' in sd:
                new_sd[f'{target}.pwconv1.weight'] = sd[f'{prefix}/pwconv1/Conv.weight']
                new_sd[f'{target}.pwconv1.bias'] = sd[f'{prefix}/pwconv1/Conv.bias']
                
            # pwconv2
            if f'{prefix}/pwconv2/Conv.weight' in sd:
                new_sd[f'{target}.pwconv2.weight'] = sd[f'{prefix}/pwconv2/Conv.weight']
                new_sd[f'{target}.pwconv2.bias'] = sd[f'{prefix}/pwconv2/Conv.bias']
            
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        print(f"Loaded vocoder weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing)}")
        # print(f"Missing: {missing}") # Uncomment to debug
        
        return missing, unexpected

    def forward(self, latents):
        """
        Args:
            latents: [Batch, 24, Latent_Length]
                     (Note: Trace mentioned 144 in latent shape, but weights 
                     strictly expect 24. If input is 144, it likely requires 
                     reshaping [B, 24, L*6] or [B, 6, 24, L] -> select before this.)
        """
        
        # 1. Un-normalize
        # Trace: Mul(std) -> Add(mean)
        x = latents * self.latent_std + self.latent_mean
        
        # 2. Embedding / Initial Conv
        x = self.input_conv(x)
        
        # 3. Backbone
        for block in self.blocks:
            x = block(x)
            
        # 4. Final Norm
        x = self.final_norm(x)
        
        # 5. Head (Projection + Shuffle)
        waveform = self.head(x)
        
        return waveform