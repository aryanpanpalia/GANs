import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

torch.manual_seed(0)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


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
