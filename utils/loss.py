import torch
import torch.nn as nn


class ComputeLoss:
    def __init__(self, g: nn.Module, d: nn.Module):
        self.g = g
        self.d = d

    def __call__(self, generated_imgs: torch.Tensor, real_imgs: torch.Tensor):
        g_imgs = generated_imgs.detach()
        d_x = torch.log()
