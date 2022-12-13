import os
import yaml

import torch
import torch.nn as nn

from model.common import Linear


def parse_model(model_yaml: dict,
                chs: list[int],
                task: str):
    layers = []
    for module in model_yaml[task]:
        f, m, n, args = module
        m = eval(m)
        if m in {Linear}:
            args = [chs[f], *args]
            chs.append(args[1])
            layers.append(*(m(*args) for _ in range(n)))
        else:
            layers.append(*(m(*args) for _ in range(n)))

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, model_yaml_path: str, img_size: int = 28, input_size: int = 100):
        super().__init__()
        self.chs = [input_size]
        with open(model_yaml_path, 'r') as f:
            model_yaml = yaml.safe_load(f)
        self.gen = parse_model(model_yaml, self.chs, 'gen')
        self.linear = nn.Linear(self.chs[-1], img_size * img_size)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.linear(self.gen(x)))


class Discriminator(nn.Module):
    def __init__(self, model_yaml_path: str, img_size: int = 28):
        super().__init__()
        self.chs = [img_size * img_size]
        with open(model_yaml_path, 'r') as f:
            model_yaml = yaml.safe_load(f)
        self.dis = parse_model(model_yaml, self.chs, 'dis')
        self.linear = nn.Linear(self.chs[-1], 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.linear(self.dis(x)))
