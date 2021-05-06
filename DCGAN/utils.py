import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import make_grid

torch.manual_seed(0)


def show_tensor_images(image_tensor, num_images=25):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)

    disc_fake_answers = disc(fake.detach())
    fake_loss = criterion(disc_fake_answers, torch.zeros_like(disc_fake_answers))

    disc_real_answers = disc(real)
    real_loss = criterion(disc_real_answers, torch.ones_like(disc_real_answers))

    return (real_loss + fake_loss) / 2


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    gen_pred = gen(noise)
    disc_pred = disc(gen_pred)
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))

    return gen_loss


def get_noise(num_images, z_dim, device):
    return torch.randn((num_images, z_dim, 1, 1)).to(device)
