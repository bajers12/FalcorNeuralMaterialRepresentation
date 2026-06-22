from typing import Tuple

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder:
      - frame extractor: Linear(C -> 6*num_frames) producing residual (Nxyz, Txyz) per frame
      - transform wi/wo into each predicted frame (T,B,N coords)
      - MLP on [z, dir_features] -> raw RGB
      - output = exp(raw - exp_offset)
    """

    def __init__(
        self,
        latent_ch: int,
        num_frames: int = 2,
        mlp_width: int = 32,
        mlp_depth: int = 2,
        use_bias_in_mlp: bool = True,
        frame_linear_bias: bool = False,
        exp_offset: float = 3.0,
    ):
        super().__init__()
        assert num_frames >= 1
        self.latent_ch = latent_ch
        self.num_frames = num_frames
        self.exp_offset = exp_offset

        self.frame_linear = nn.Linear(latent_ch, 6 * num_frames, bias=frame_linear_bias)

        mlp_in = latent_ch + 6 * num_frames
        layers = []
        prev = mlp_in
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width, bias=use_bias_in_mlp))
            layers.append(nn.ReLU(inplace=True))
            prev = mlp_width
        layers.append(nn.Linear(prev, 3, bias=use_bias_in_mlp))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        out = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def loss(
        self,
        raw: torch.Tensor,
        y: torch.Tensor,
        exp_offset: float,
        eps: float,
        mask_threshold: float = 1e-4,
    ) -> torch.Tensor:
        return Decoder.log_l1_loss(raw, y, exp_offset, eps, mask_threshold)

    def _predict_frames(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z [B,C] -> T, Bv, N  each [B, num_frames, 3]
        """
        Bsz = z.shape[0]
        ft = self.frame_linear(z).view(Bsz, self.num_frames, 6)

        # T is intentionally not orthogonalized before forming B = cross(N, T).
        n_raw = ft[..., 0:3].clone()
        t_raw = ft[..., 3:6].clone()
        n_raw[..., 2] = n_raw[..., 2] + 1.0
        t_raw[..., 0] = t_raw[..., 0] + 1.0
        N = self._safe_normalize(n_raw)  # [B, F, 3]
        T = self._safe_normalize(t_raw)  # [B, F, 3]

        Bv = torch.cross(N, T, dim=-1)  # [B, F, 3]
        Bv = self._safe_normalize(torch.cross(N, T, dim=-1))

        return T, Bv, N

    def forward_raw(
        self, z: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
    ) -> torch.Tensor:
        """
        z:  [B,C]
        wi: [B,3] (local)
        wo: [B,3] (local)
        returns raw RGB logits before exp parameterization
        """
        T, Bv, N = self._predict_frames(z)

        wi_f = torch.stack(
            [
                (wi.unsqueeze(1) * T).sum(dim=-1),
                (wi.unsqueeze(1) * Bv).sum(dim=-1),
                (wi.unsqueeze(1) * N).sum(dim=-1),
            ],
            dim=-1,
        )
        wo_f = torch.stack(
            [
                (wo.unsqueeze(1) * T).sum(dim=-1),
                (wo.unsqueeze(1) * Bv).sum(dim=-1),
                (wo.unsqueeze(1) * N).sum(dim=-1),
            ],
            dim=-1,
        )

        dir_feats = torch.cat([wi_f, wo_f], dim=-1).view(
            z.shape[0], 6 * self.num_frames
        )
        x = torch.cat([z, dir_feats], dim=-1)
        return self.mlp(x)

    def forward(
        self, z: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
    ) -> torch.Tensor:
        raw = self.forward_raw(z, wi, wo)
        return torch.exp(raw - self.exp_offset)

    @staticmethod
    def log_diff(
        raw: torch.Tensor,
        y: torch.Tensor,
        exp_offset: float,
        eps: float,
        mask_threshold: float = 1e-4,
    ) -> torch.Tensor:
        y = torch.nan_to_num(y, nan=0.0, posinf=1e30, neginf=0.0).clamp_min(0.0)

        # Build per-sample mask: keep samples that have at least one significant channel
        valid = y.amax(dim=-1) >= mask_threshold  # [B]
        if valid.any():
            raw_c = raw[valid]
            y_c = y[valid].clamp_min(eps)
        else:
            # Fallback: use everything (avoids zero-element mean on pathological batches)
            raw_c = raw
            y_c = y.clamp_min(eps)

        return ((raw_c - exp_offset) - torch.log(y_c))

    @staticmethod
    def log_l1_loss(
        raw: torch.Tensor,
        y: torch.Tensor,
        exp_offset: float,
        eps: float,
        mask_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """
        L1 loss in log space:
        mean(|(raw - exp_offset) - log(y+eps)|)

        This is equivalent to taking the log of the exponential decoder output, but
        avoids overflowing exp(raw - exp_offset) before the logarithm is applied.
        """
        return Decoder.log_diff(raw, y, exp_offset, eps, mask_threshold).abs().mean()

    @staticmethod
    def log_l2_loss(
        raw: torch.Tensor,
        y: torch.Tensor,
        exp_offset: float,
        eps: float,
        mask_threshold: float = 1e-4,
    ) -> torch.Tensor:
        return Decoder.log_diff(raw, y, exp_offset, eps, mask_threshold).square().mean()
