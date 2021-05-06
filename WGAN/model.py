import torch
from torch import nn

torch.manual_seed(0)


class Generator(nn.Module):
    def __init__(self, z_dim=100, hidden_dim=64, im_chan=1):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self._get_block(z_dim, hidden_dim * 16, 4, 1, 0),
            self._get_block(hidden_dim * 16, hidden_dim * 8, 4, 2, 1),
            self._get_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            self._get_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.ConvTranspose2d(hidden_dim * 2, im_chan, 4, 2, 1),
            nn.Tanh()
        )

    @staticmethod
    def _get_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, noise):
        return self.gen(noise)


class Critic(nn.Module):
    def __init__(self, im_channels, hidden_dim):
        super(Critic, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(im_channels, hidden_dim * 1, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._get_block(hidden_dim * 1, hidden_dim * 2, 4, 2, 1),
            self._get_block(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            self._get_block(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.Conv2d(hidden_dim * 8, 1, 4, 2)
        )

    @staticmethod
    def _get_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        return self.disc(image)
