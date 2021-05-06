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
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
EPOCHS = 5
HIDDEN_DIM = 64
CRITIC_ITERATIONS = 5
LAMBDA = 10

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
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))
initialize_weights(gen)
gen.train()

critic = Critic(im_channels=IMG_CHANNELS, hidden_dim=HIDDEN_DIM)
critic_opt = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))
initialize_weights(critic)
critic.train()

step = 0
for epoch in range(EPOCHS):
    for reals, _ in tqdm(dataloader):
        reals = reals.to(DEVICE)

        # Train Discriminator
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((len(reals), Z_DIM, 1, 1)).to(DEVICE)
            fakes = gen(noise)
            fake_preds = critic(fakes)
            real_preds = critic(reals)
            gp = gradient_penalty(critic, reals, fakes, device=DEVICE)
            critic_loss = torch.mean(fake_preds) - torch.mean(real_preds) + (LAMBDA * gp)

            critic_opt.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_opt.step()

        noise = torch.randn((len(reals), Z_DIM, 1, 1)).to(DEVICE)
        fakes = gen(noise)
        fake_preds = critic(fakes)
        gen_loss = -torch.mean(fake_preds)

        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                show_tensor_images(fakes)
                show_tensor_images(reals)

        step += 1
