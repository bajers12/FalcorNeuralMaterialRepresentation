import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTexture(nn.Module):
    """
    Learnable latent texture pyramid.
    Each level is an independent [1, C, H, W] tensor, sampled with bilinear
    filtering using uv in [0,1].
    """

    def __init__(self, h: int, w: int, c: int, mip_count: int = 1, init_std: float = 0.01):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c
        self.mip_count = max(1, int(mip_count))
        levels = []
        for level in range(self.mip_count):
            level_h = max(1, h >> level)
            level_w = max(1, w >> level)
            levels.append(nn.Parameter(torch.randn(1, c, level_h, level_w) * init_std))
        self.levels = nn.ParameterList(levels)

    @property
    def Z(self) -> nn.Parameter:
        return self.levels[0]

    def level_shape(self, level: int) -> tuple[int, int]:
        z = self.levels[level]
        return int(z.shape[-2]), int(z.shape[-1])

    def _sample_level(self, level: int, uv: torch.Tensor) -> torch.Tensor:
        if uv.numel() == 0:
            return uv.new_empty((0, self.c))
        grid = (uv * 2.0 - 1.0).view(1, -1, 1, 2)  # [1,B,1,2]
        z = F.grid_sample(
            self.levels[level],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [1,C,B,1]
        return z.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # [B,C]

    def sample(self, uv: torch.Tensor, mip_level: torch.Tensor | None = None) -> torch.Tensor:
        """
        uv: [B,2] in [0,1]
        mip_level: optional [B] tensor containing integer mip levels
        returns z: [B,C]

        Important: keep the latent texture batch dimension at 1.
        Expanding self.Z to [B,C,H,W] makes autograd build a gradient of that
        size during backward, which explodes memory for large B.
        """
        if mip_level is None or self.mip_count == 1:
            return self._sample_level(0, uv)

        mip = mip_level.to(device=uv.device).round().long().clamp(0, self.mip_count - 1).view(-1)
        out = uv.new_empty((uv.shape[0], self.c))
        for level in range(self.mip_count):
            mask = mip == level
            if mask.any():
                out[mask] = self._sample_level(level, uv[mask])
        return out
