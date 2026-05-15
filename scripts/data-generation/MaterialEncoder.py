import torch
import torch.nn as nn


class MaterialEncoder(nn.Module):
    """
    Training-only encoder that maps sampled material parameters to latent codes.
    The runtime path still only consumes the baked latent texture.
    """

    def __init__(self, input_ch: int, latent_ch: int, hidden_width: int = 32, depth: int = 2):
        super().__init__()
        layers = []
        prev = input_ch
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden_width))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_width
        layers.append(nn.Linear(prev, latent_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
