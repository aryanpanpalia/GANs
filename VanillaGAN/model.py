import torch
from torch import nn

torch.manual_seed(0)


class Generator(nn.Module):
    def __init__(self, z_dim=10, hidden_dim=128, im_dim=784):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.get_gen_block(z_dim, hidden_dim),
            self.get_gen_block(hidden_dim, hidden_dim * 2),
            self.get_gen_block(hidden_dim * 2, hidden_dim * 4),
            self.get_gen_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    @staticmethod
    def get_gen_block(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def get_discriminator_block(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc
