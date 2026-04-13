import torch
import torch.nn as nn
import torch.optim as optim

def build_mlp(input_dim, output_dim, depth, width, batch_norm=False):
    layers = [nn.Linear(input_dim, width)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(width))
    layers.append(nn.LeakyReLU())
    for _ in range(depth - 1):
        layers.append(nn.Linear(width, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(width, output_dim))

    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            nn.init.zeros_(m.bias)
    return model