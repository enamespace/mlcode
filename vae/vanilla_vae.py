from torch import nn
import torch
from torchsummary import summary
# 针对手写数字


class VAE(nn.Module):
    def __init__(self, in_channel,  latent_dim=10, hiddem_dims=None,) -> None:
        super().__init__()
        if hiddem_dims is None:
            hiddem_dims = [32, 64, 128, 256, 512]
        dim_list = [in_channel] + hiddem_dims
        in_out = list(zip(dim_list[:-1], dim_list[1:]))

        print(in_out)

        in_modules = []

        for in_dim, out_dim in in_out:
            in_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(num_features=out_dim),
                    nn.LeakyReLU()
                )
            )

        self.encoder = nn.Sequential(*in_modules)

        # 乘以4是因为stride = 2, H W 不断减小了一半，最后为2*2
        self.fc_mean = nn.Linear(hiddem_dims[-1] * 4, latent_dim)
        self.fc_log_var = nn.Linear(hiddem_dims[-1] * 4, latent_dim)

        out_modules = []
        in_out.reverse()
        print(in_out)
        for in_dim, out_dim in in_out[:-1]:
            out_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=out_dim,
                        out_channels=in_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(num_features=in_dim),
                    nn.LeakyReLU()
                )

            )

        self.decoder = nn.Sequential(*out_modules)
        self.decoder_input = nn.Linear(latent_dim, hiddem_dims[-1] * 4)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hiddem_dims[0],
                hiddem_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(num_features=hiddem_dims[0]),
            nn.LeakyReLU(),
            nn.Conv2d(hiddem_dims[0], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # encode result
        encode_result = self.encoder(x)
        encode_result = torch.flatten(encode_result, start_dim=1)

        mean_v = self.fc_mean(encode_result)
        log_var_v = self.fc_log_var(encode_result)

        z = mean_v + torch.exp(log_var_v * 0.5) * torch.rand_like(log_var_v)

        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 512, 2, 2)
        result = self.decoder(decoder_input)
        print(result.shape)
        recons = self.final_layer(result)
        return [recons, mean_v, log_var_v]

    def loss(self, input, recons, mu, log_var, kld_weight):
        recons_loss = nn.functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}


x = torch.randn(16, 3, 64, 64)

model = VAE(3, 10)
print(summary(model, (3, 64, 64), device='cpu'))
recons, mean_v, log_var_v = model(x)
loss = model.loss(x, recons, mean_v, log_var_v, 0.005)
print(loss)
