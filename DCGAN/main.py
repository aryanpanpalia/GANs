import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

sys.path.append(".")

from model import *
from utils import *

torch.manual_seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
NUM_EPOCHS = 5
HIDDEN_DIM = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]),
    ]
)

dataloader = DataLoader(
    MNIST(root='dataset/', transform=transforms, download=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

gen = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, im_chan=IMG_CHANNELS)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
initialize_weights(gen)
gen.train()

disc = Discriminator(im_channels=IMG_CHANNELS, hidden_dim=HIDDEN_DIM)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
initialize_weights(disc)
disc.train()

criterion = torch.nn.BCELoss()

step = 0
for epoch in range(NUM_EPOCHS):
    for reals, _ in tqdm(dataloader):
        cur_batch_size = len(reals)
        reals = reals.to(DEVICE)

        # Train Discriminator
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, reals, cur_batch_size, Z_DIM, DEVICE)
        disc_loss.backward()
        disc_opt.step()

        # Train Generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, Z_DIM, DEVICE)
        gen_loss.backward()
        gen_opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                noise = get_noise(cur_batch_size, Z_DIM, DEVICE)
                fakes = gen(noise)
                show_tensor_images(fakes)
                show_tensor_images(reals)

        step += 1
