import torch
from torch import nn

from einops import einsum, rearrange

class SpatialTransformer(nn.Module):
    def __init__(self, channels, d_cond, n_heads, layers) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.transformer_block = nn.ModuleList(
            [
                BasicTransformer(channels, d_cond, channels, n_heads)
                for _ in range(layers)
            ]
        )  

        self.proj_out = nn.Conv2d(channels, channels)

    # [B, C, H, W]
    def forward(self, x, cond):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2,3,1).view(b, h*w, c)
        for block in self.transformer_block:
            x = block(x, cond)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)

        return x + x_in


class BasicTransformer(nn.Module):
    def __init__(self,in_dim,  d_cond, attn_dim, n_heads=8) -> None:
        super().__init__()
        self.attn1 = CrossAttention(in_dim,  d_cond, attn_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn2 = CrossAttention(in_dim,  d_cond, attn_dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.Dropout(),
            nn.Linear(in_dim * 4, in_dim)
        )
        self.norm3 = nn.LayerNorm(in_dim)

    
    def forward(self, x, cond):

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond) + x
        x = self.ffn(self.nomr3(x)) + x
        return x
        

        

class CrossAttention(nn.Module):

    def __init__(self, in_dim, d_cond, attn_dim, n_heads=8) -> None:
        super().__init__()
        self.attn_dim = attn_dim


        self.q = nn.Linear(in_dim, attn_dim)
        self.k = nn.Linear(d_cond, attn_dim)
        self.v = nn.Linear(d_cond, attn_dim)
        self.out = nn.Linear(attn_dim, in_dim)
        self.heads = n_heads
        self.scale = (attn_dim / n_heads) ** 0.5

    def forward(self, x, cond=None):
        # [B * L * C]
        if cond is None:
            cond = x
        q = self.q(x)
        k = self.k(cond)
        v = self.v(cond)
        qkv = [q, k, v]

        b, c, h, w = q.shape

        from einops import rearrange, einsum

        q, k, v = map(
            lambda t: rearrange(t, "b l (n h) -> b n l h", n=self.heads),
            qkv
        )

        q = q * self.scale
        scores = einsum(q, k, "b n i h, b n j h  -> b n i j")
        scores = torch.softmax(scores, dim=-1)
        result = einsum(scores, v, "b n i j, b n j h -> b n i d")
        result = rearrange(result, "b n l d -> b l (n d)",)

        return self.out(result)
    
class Unet(nn.Module):

    def __init__(self,  input_dim, dim_factors=[1, 2, 4, 8]):
        self.input_dim = input_dim

    def forward(self, x, cond,):
        pass


class StableDiffusion(nn.Module):

    def __init__(self, 
                 autoencoder, 
                 unet_model, 
                 cond_encoder, 
                 latent_scalor, 
                 steps,
                 linear_start,
                 linear_end):
        super().__init__()

        self.model = unet_model
        self.first_stage_model = autoencoder
        self.cond_stage_model = cond_encoder
        self.latent_scalor = latent_scalor
        
        beta = torch.linspace(linear_start, linear_end, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha)
        self.alpha_bar = alpha_bar

    def forward(self, x, t, cond):
        return self.model(x, t, cond)
