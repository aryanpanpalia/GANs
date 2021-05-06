import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
import sys

sys.path.append(".")

from model import Generator, Discriminator
from utils import *

torch.manual_seed(0)

# parameters
CRITERION = nn.BCEWithLogitsLoss()
NUM_EPOCHS = 200
Z_DIM = 64
DISPLAY_STEP = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.00001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=True
)

gen = Generator(Z_DIM).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
disc = Discriminator().to(DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(NUM_EPOCHS):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(DEVICE)

        # update discriminator
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, CRITERION, real, cur_batch_size, Z_DIM, DEVICE)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # update generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, CRITERION, cur_batch_size, Z_DIM, DEVICE)
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        # Keep track of the average loss
        mean_discriminator_loss += disc_loss.item() / DISPLAY_STEP
        mean_generator_loss += gen_loss.item() / DISPLAY_STEP

        # Visualization code
        if cur_step % DISPLAY_STEP == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}"
            )
            fake_noise = get_noise(cur_batch_size, Z_DIM, device=DEVICE)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0

        cur_step += 1
