import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTexture(nn.Module):
    """
    Learnable latent texture Z of shape [1, C, H, W].
    Sampled with bilinear filtering using uv in [0,1].
    """

    def __init__(self, h: int, w: int, c: int, init_std: float = 0.01):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c
        z = torch.randn(1, c, h, w) * init_std
        self.Z = nn.Parameter(z)

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: [B,2] in [0,1]
        returns z: [B,C]

        Important: keep the latent texture batch dimension at 1.
        Expanding self.Z to [B,C,H,W] makes autograd build a gradient of that
        size during backward, which explodes memory for large B.
        """
        grid = (uv * 2.0 - 1.0).view(1, -1, 1, 2)  # [1,B,1,2]
        z = F.grid_sample(
            self.Z,  # [1,C,H,W]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [1,C,B,1]
        return z.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # [B,C]
