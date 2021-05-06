import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid


def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    mixed = real * epsilon + fake * (1 - epsilon)

    mixed_preds = critic(mixed)
    gradient = torch.autograd.grad(
        inputs=mixed,
        outputs=mixed_preds,
        grad_outputs=torch.ones_like(mixed_preds),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


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
