import os
import sys
from pathlib import Path
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.dataloader import MyDataset
from model.gan import Generator, Discriminator

EPOCHS: int = 10000
BATCH_SIZE: int = 2
lEARNING_RATE: float = 0.01
MODEL_YAML_PATH: str = 'model/gan.yaml'
DATA_PATH: str = 'd:\\Data\\GAN\\partial'

criterion = nn.BCELoss()


def compute_d_loss(d_x: torch.Tensor, d_g_x: torch.Tensor):
    batch_size = d_x.shape[0]
    assert d_x.shape[0] == d_g_x.shape[0], 'batch size is wrong'

    return torch.sum(-torch.log(d_x) - torch.log(1 - d_g_x)) / batch_size


def compute_g_loss(d_g_x: torch.Tensor):
    return torch.sum(-torch.log(1 - d_g_x)) / d_g_x.shape[0]


def main():
    ds = MyDataset(DATA_PATH, 28)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # pbar = enumerate(train_loader)
    nb = len(train_loader)

    real_targets = torch.ones((BATCH_SIZE, 1))
    fake_targets = torch.zeros((BATCH_SIZE, 1))

    g = Generator(MODEL_YAML_PATH)
    d = Discriminator(MODEL_YAML_PATH)

    g_optim = SGD(g.parameters(), lr=0.001, momentum=0.99)
    d_optim = SGD(d.parameters(), lr=0.001, momentum=0.99)

    cnt = 0

    for epoch in range(EPOCHS):
        # print(epoch)
        # for ni, (imgs, zs) in pbar:
        for imgs, zs in train_loader:
            # print(imgs.shape)
            # print(zs.shape)
            imgs = imgs.float()
            generated_imgs = g(zs).detach()
            d_x = d(imgs)
            d_g_x = d(generated_imgs)

            d_optim.zero_grad()
            g_optim.zero_grad()

            d_real_loss = criterion(d_x, real_targets)
            d_fake_loss = criterion(d_g_x, fake_targets)

            d_loss = d_fake_loss + d_fake_loss

            d_real_loss.backward()
            d_fake_loss.backward()

            d_optim.step()

            generated_imgs = g(zs)
            d_g_x = d(generated_imgs)
            g_loss = criterion(d_g_x, real_targets)

            g_loss.backward()

            g_optim.step()

            print(f'd loss = {d_loss} g loss = {g_loss}')

    # inputs = torch.randn((4, 100))
    # fake_imgs = g(inputs)

    torch.save(g.state_dict(), 'g.pt')


if __name__ == '__main__':
    main()
