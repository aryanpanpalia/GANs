import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)


def show_tensor_images(image_tensor, num_images=25, nrow=5):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    return torch.cat((x, y), 1).float()


def get_input_dimensions(z_dim, image_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = n_classes + image_shape[0]

    return generator_input_dim, discriminator_im_chan


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
