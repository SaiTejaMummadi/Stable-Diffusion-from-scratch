import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        #(1,1280)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self,in_channels:int, out_channels:int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identiy()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        #Featrues: bsize, inchannels, height, widht
        #time (1,1280)
        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)
  
        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merge = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd:int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        #bsize, features, height, width
        #context: bsize, seqlen, dim

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n,c,h,w = x.shape

        #Bsize, features, height, width -> bsize, features, h*w
        x = x.view((n,c,h*w))

        #Bsize, features, h*w -> bsize, h*w, features

        x = x.transpose(-1,-2)
        #Normalization + Self Attention with skip connection

        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short
        residue_short =x

        #Normalization + Cross Attention with skip connection
        x = self.layernorm_2(x)

        self.attention_2(x)

        x += residue_short
        residue_short =x

        #Normalization + FF with GeGLU and skip connection
        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2,dim=-1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        #bsize, h*w, features -> bsize, features, h*w
        x = x.transpose(-1,-2)

        x = x.view((n,c,h,w))

        return self.conv_output(x) + residue_long




class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x,context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x,time)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3, padding=1)
    
    def forward(self, x):
        #Bsize, features, height, width -> bsize, features, height*2, width*2
        x = F.interpolate(x,scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([
            #Bsize, 4, height/8, width/8
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),

            #Bsize, 320, height/8, width/8 -> #Bsize, 320, height/16, width/16
            SwitchSequential(nn.Conv2d(320,320,kernel_size3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,640),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,640),UNET_AttentionBlock(8,80)),

            #Bsize, 640, height/16, width/16 -> #Bsize, 640, height/32, width/32
            SwitchSequential(nn.Conv2d(640,640,kernel_size3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280),UNET_AttentionBlock(8,160)),

            #Bsize, 1280, height/32, width/32 -> #Bsize, 1280, height/64, width/64
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

        ])

        self.bottlenext = SwitchSequential(
            UNET_ResidualBlock(1280,1280),

            UNET_AttentionBlock(8,160),

            UNET_ResidualBlock(1280,1280),

        )

        self.decoders = nn.ModuleList([
            # bsize, 2560, height/64, width/64 -> bsize, 1280, height/64, width/64
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),            
            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8,160),Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920,640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8,80),Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,40)),
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        #x : Bsize, 320, height/8, width/8
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)
    
    def forward(self, latent:torch.Tensor, context:torch.Tensor, time: torch.Tensor):
        #latent: bsize, 4, height/8, width/8
        #context: bsize, seqlen, dim
        #time: (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)

        #(bsize, 4, height / 8, width/8) -> bsize, 320, height/8, width/8
        output = self.unet(latent, context, time)

        #bsize, 320, height/8, width/8 -> bsize, 4, height/8, width/8
        output = self.final(output)

        return output

