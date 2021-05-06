import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

sys.path.append(".")

from model import *
from utils import *

torch.manual_seed(0)

IMAGE_SHAPE = (1, 28, 28)
NUM_CLASSES = 10
CRITERION = nn.BCEWithLogitsLoss()
NUM_EPOCHS = 200
Z_DIM = 64
DISPLAY_STEP = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

generator_input_dim, discriminator_im_chan = get_input_dimensions(Z_DIM, IMAGE_SHAPE, NUM_CLASSES)

gen = Generator(input_dim=generator_input_dim).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
disc = Discriminator(im_chan=discriminator_im_chan).to(DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

cur_step = 0
generator_losses = []
discriminator_losses = []

for epoch in range(NUM_EPOCHS):
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(DEVICE)

        one_hot_labels = get_one_hot_labels(labels.to(DEVICE), NUM_CLASSES)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, IMAGE_SHAPE[1], IMAGE_SHAPE[2])

        # Update discriminator
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, Z_DIM, device=DEVICE)

        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels).detach()
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)

        disc_fake_pred = disc(fake_image_and_labels)
        disc_real_pred = disc(real_image_and_labels)

        disc_fake_loss = CRITERION(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = CRITERION(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        # Update generator
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = CRITERION(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]

        if cur_step % DISPLAY_STEP == 0:
            gen_mean = sum(generator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
            disc_mean = sum(discriminator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
            print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1
