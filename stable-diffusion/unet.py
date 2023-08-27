import torch
from torch import nn
from ldm import SpatialTransformer



class Downsample(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.op = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    def forward(self, x):
        return self.op(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.op = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels, d_t_emb, out_channels) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.in_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
        )

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if channels != out_channels:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, t_emb):

        h = self.in_layers(x)
        t_emb = self.embed_layers(t_emb).type(h.type)
        h = h + t_emb[:, :, None, None]

        h = self.out_layers(h)

        return self.skip_connection(x) + h


class UniveralBlock():
    pass


class Unet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 channels,
                 n_res_block,
                 attn_levels,
                 channel_mults,
                 n_heads,
                 tf_layers,
                 d_cond
                 ) -> None:
        super().__init__()
        d_time_emb = 4 * channels
        self.time_emb = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb)
        )

        self.input_blocks = nn.ModuleList()

        self.input_blocks.append(UniveralBlock(
            nn.Conv2d(in_channels, channels)))

        levels = len(channel_mults)
        channels_list = [channels * m for m in channel_mults]


        # down
        for i in range(levels):
            for _ in range(n_res_block):

                layers = [
                    ResBlock(channels, d_time_emb, channels_list[i])]

                if i in attn_levels:
                    layers.append(SpatialTransformer(channels_list[i], d_cond, n_heads, tf_layers))

                self.input_blocks.append(UniveralBlock(*layers))

                channels = channels_list[i]

            if i !=  levels - 1:
                self.input_blocks.append(Downsample(channels))

        # mid

        self.mid_block = UniveralBlock(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, d_cond, n_heads, tf_layers),
            ResBlock(channels, d_time_emb),
        )


        # up
        self.output_blocks = []
        for i in reversed(range(levels)):
            for j in range(n_res_block + 1):

                layers = [ResBlock(channels + channels_list[i], d_time_emb, channels_list[i])]

                if i in attn_levels:
                    layers.append(SpatialTransformer(channels_list[i], d_cond, n_heads, tf_layers))

                self.output_blocks.append(UniveralBlock(*layers))
                channels = channels_list[i]
                if i != 0 and j == n_res_block:
                    layers.append(Upsample(channels))
           
                

        
        self.out = nn.Sequential(
            nn.Conv2d(channels, out_channels, 3, padding=1)
        )


    def forward(self, x, t, cond):
        
        x_input = []
        t_emb = self.time_emb(t)

        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input.append(x)


        self.mid_block(x, t_emb, cond)


        for module in self.output_blocks:
            x = module(torch.cat(x + x_input.pop(), dim = 1), t_emb, cond)


        return self.out(x)
