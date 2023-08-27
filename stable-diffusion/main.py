from torch import nn
import torch

from unet import Unet

from ldm import StableDiffusion

def txt2img():


    in_channels = out_channels = 4
    channel_mults = [1, 2, 4, 8]
    n_heads = 8
    tf_layers = 1
    attn_levels = [0, 1, 2]
    d_cond = 768

    unet_model = Unet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=320,
        attn_levels=attn_levels,
        channel_mults=channel_mults,
        n_heads=n_heads,
        tf_layers=tf_layers,
        d_cond=d_cond
    )

    sd = StableDiffusion(
        autoencoder=None,
        unet_model=unet_model,
        cond_encoder=None,
        latent_scalor=8,
        steps=40,
        linear_start=0.1,
        linear_end=0.2
    )

    # user input prompts, t
    prompts = []
    
    cond = self.model.get_text_cond(prompts)

    # 从模型采样eps
    self.sampler = wrapper(sd)
    # unet
    x = self.sampler.sample(cond, shape,)


    image = self.model.autoencoder.decode(x)

    
    