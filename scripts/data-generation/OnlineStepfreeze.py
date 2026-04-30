#!/usr/bin/env python3
"""
Neural Material MVP Trainer (PyTorch) — with optional normals
------------------------------------------------------------
Trains:
  - a learnable latent texture Z(u,v) with C channels (default 8)
  - a decoder:
      * frame extractor: Linear(C -> 12) to predict 2 shading frames (N,T) each
      * direction transform of wi/wo into each predicted frame (T,B,N)
      * MLP: ReLU hidden layers -> 3 outputs -> exp(out - offset)

Targets:
  Assumes dataset provides targets y = f(wi,wo) * max(0, n·wo)  (cos baked in).

Dataset formats supported:
  - .npz containing arrays:
      required: 'uv', 'wi', 'wo', and ('y' or 'rgb')
      optional: 'normal'

Normals support:
  Sampled normals are treated as training-side material information and are not used
  to rotate wi/wo. The decoder always sees the original sampled directions.

Exports:
    - latent_texture.pt: {"Z": [1,C,H,W], "shape": (H,W,C)}
    - latent_rgba0.npz / latent_rgba1.npz if C==8 (for renderer-friendly RGBA splits)
    - latent0.exr / latent1.exr if C==8 (renderer-ready assets)
    - decoder.pt: PyTorch state_dict
    - decoder_weights.npz: Numpy arrays for renderer-side loading
    - decoder_weights.bin / metadata.json: renderer-ready bundle for NeuralMaterial
"""

from __future__ import annotations

import os
import math
import json
import time
import argparse
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional
from DataGenerator import (
    DataGenerator,
    SEED_DOMAIN_BOOTSTRAP,
    SEED_DOMAIN_TRAIN,
    SEED_DOMAIN_VALIDATION,
)
from training_run_logging import TrainingRunLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================


@dataclass
class TrainConfig:
    # Latent texture
    tex_h: int = 512
    tex_w: int = 512
    latent_ch: int = 8

    # Decoder architecture
    num_frames: int = 2
    mlp_width: int = 32
    mlp_depth: int = 2  # number of hidden layers
    use_bias_in_mlp: bool = True
    frame_linear_bias: bool = False

    # Importance sampling decoder architecture
    sampler_mlp_width: int = 32
    sampler_mlp_depth: int = 2

    # Output parameterization
    exp_offset: float = 3.0
    clamp_min_target: float = 0.0  # safety clamp on y before log
    log_eps: float = 1e-6  # y' = clamp(y, eps) for log

    # Optimization
    device: str = "cuda"
    seed: int = 1337
    training_n: int = 65536 # total samples generated per outer epoch
    validation_n: int = 65536
    max_epochs: int = 50

    lr: float = 1e-3
    lr_min: float = 1e-4
    lr_latent: Optional[float] = None
    lr_decoder: Optional[float] = None
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None

    # Logging / checkpoints
    out_dir: str = "./output_weights"
    preview_out_dir: str = ""
    print_every_epochs: int = 10000

    # Training behavior
    train_latent_texture: bool = True
    train_decoder: bool = True
    freeze_latent_after_epoch: Optional[int] = None
    freeze_decoder_after_epoch: Optional[int] = None

    # Legacy flag retained for CLI compatibility. Direction transforms are disabled;
    # sampled normals remain available only as training-side material features.
    use_normals: bool = False

    # Training-only encoder that maps sampled material values to latent codes.
    encoder_width: int = 32
    encoder_depth: int = 2
    encoder_bootstrap_epochs: int = 200
    latent_init_batch_size: int = 65536
    use_albedo_features: bool = True
    use_spec_features: bool = True
    use_normal_features: bool = True
    use_roughness_feature: bool = True
    use_pdf_feature: bool = False

    # Importance sampling decoder configuration
    train_importance_sampler: bool = True

# =============================================================================
# Batch handling
# =============================================================================


def tensorize_batch(data_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value).float() if isinstance(value, np.ndarray) else value
        for key, value in data_dict.items()
    }


def print_first_sample(batch: Dict[str, torch.Tensor], label: str) -> None:
    print(f"[debug] first sample from {label}:")
    ordered_keys = ["uv", "wi", "wo", "y", "spec", "albedo", "normal", "roughness", "pdf"]
    for key in ordered_keys:
        if key not in batch:
            continue
        value = batch[key][0]
        if value.ndim == 0:
            print(f"  {key}: {value.item():.6f}")
        else:
            flat = value.detach().cpu().tolist()
            formatted = ", ".join(f"{float(x):.6f}" for x in flat)
            print(f"  {key}: [{formatted}]")


def get_encoder_input_dim(cfg: TrainConfig) -> int:
    dim = 0
    if cfg.use_albedo_features:
        dim += 3
    if cfg.use_spec_features:
        dim += 3
    if cfg.use_normal_features:
        dim += 3
    if cfg.use_roughness_feature:
        dim += 1
    if cfg.use_pdf_feature:
        dim += 1
    if dim == 0:
        raise ValueError("Encoder feature set is empty. Enable at least one training feature.")
    return dim


# =============================================================================
# Model
# =============================================================================


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


class Decoder(nn.Module):
    """
    Decoder:
      - frame extractor: Linear(C -> 6*num_frames) producing (Nxyz, Txyz) per frame
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

    def _predict_frames(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z [B,C] -> T, Bv, N  each [B, num_frames, 3]
        """
        Bsz = z.shape[0]
        ft = self.frame_linear(z).view(Bsz, self.num_frames, 6)

        N = self._safe_normalize(ft[..., 0:3])  # [B, F, 3]

        # Re-orthogonalise T against N (Gram-Schmidt), then normalise
        T_raw = ft[..., 3:6]  # [B, F, 3]
        T_raw = T_raw - (T_raw * N).sum(dim=-1, keepdim=True) * N
        T = self._safe_normalize(T_raw)  # [B, F, 3]

        # Bitangent: N × T is already unit length because N and T are orthonormal
        Bv = torch.cross(N, T, dim=-1)  # [B, F, 3]

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


class ImportanceSamplingDecoder(nn.Module):
    """
    Importance sampling decoder matching the paper (Section 4.3 / Figure 4):
      - The network predicts parameters of a two-lobe anisotropic GGX (Trowbridge-Reitz)
        microfacet distribution, NOT a von Mises-Fisher distribution.
      - Each lobe has: anisotropic roughness (alpha_x, alpha_y), lobe weight, and its own
        learned shading frame (T, B, N) extracted from the latent code.
      - Sampling follows the standard GGX visible-normal (VNDF) path:
          1. Sample a microfacet normal m ~ GGX-NDF in the lobe's shading frame.
          2. Reflect wi around m to get wo.
          3. Evaluate the blended two-lobe PDF at wo.
      - The MLP only sees wi (not wo), consistent with Fig. 4 ("ωi" feeds the sampler).

    Network outputs (per forward pass, shape [B, 2*5]):
        For each lobe i in {0, 1}:
            raw_alpha_x_i, raw_alpha_y_i  -> alpha = alpha_min + (alpha_max-alpha_min)*sigmoid(·)
            raw_weight_i                  -> lobe weight (softmax across lobes)
            (frame comes from frame_linear, shared with BRDF decoder convention)

    The frame_linear layer still outputs 6*num_frames values so that the weight layout
    is identical to the BRDF Decoder and can share the same export path.
    num_frames must equal 2 (one frame per lobe).
    """

    # Roughness is clamped to [alpha_min, 1] to avoid singularities in the GGX NDF.
    ALPHA_MIN: float = 0.01
    ALPHA_MAX: float = 1.0

    def __init__(
        self,
        latent_ch: int,
        num_frames: int = 2,
        mlp_width: int = 32,
        mlp_depth: int = 2,
        use_bias_in_mlp: bool = True,
        frame_linear_bias: bool = False,
    ):
        super().__init__()
        if num_frames != 2:
            raise ValueError(
                "ImportanceSamplingDecoder requires num_frames=2 "
                "(one GGX lobe per shading frame)."
            )
        self.latent_ch = latent_ch
        self.num_frames = num_frames  # == 2

        # Frame extractor: same layout as Decoder — 6*num_frames outputs (N_i, T_i per frame).
        self.frame_linear = nn.Linear(latent_ch, 6 * num_frames, bias=frame_linear_bias)

        # MLP inputs: latent z  +  wi projected into each frame (3 per frame)
        # MLP outputs: 5 values per lobe × 2 lobes = 10 raw scalars
        #   per lobe: [raw_alpha_x, raw_alpha_y, raw_weight] — 3 values × 2 lobes = 6
        #   but we also output 2 extra values (one per lobe) reserved for future
        #   anisotropy-axis control; set to zero for now so the output dim stays
        #   at 2*5=10 to match the paper's Fig. 4 description of a "two-lobe distribution".
        #
        # Concretely the 10 outputs are laid out as:
        #   [lobe0: raw_ax, raw_ay, raw_w, _, _,   lobe1: raw_ax, raw_ay, raw_w, _, _]
        # The two unused slots (_) give the MLP room to express per-lobe properties
        # without changing the export format when extended later.
        mlp_in = latent_ch + 3 * num_frames
        layers: list[nn.Module] = []
        prev = mlp_in
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width, bias=use_bias_in_mlp))
            layers.append(nn.ReLU(inplace=True))
            prev = mlp_width
        layers.append(nn.Linear(prev, 5 * num_frames, bias=use_bias_in_mlp))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        out = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict_frames(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Identical Gram-Schmidt construction as Decoder._predict_frames.
        z [B,C] -> T, Bv, N  each [B, num_frames, 3]
        """
        bsz = z.shape[0]
        ft = self.frame_linear(z).view(bsz, self.num_frames, 6)

        n = self._safe_normalize(ft[..., 0:3])
        t_raw = ft[..., 3:6]
        t_raw = t_raw - (t_raw * n).sum(dim=-1, keepdim=True) * n
        t = self._safe_normalize(t_raw)
        bv = torch.cross(n, t, dim=-1)
        return t, bv, n  # each [B, num_frames, 3]

    @staticmethod
    def _ggx_ndf(m_local: torch.Tensor,
                 alpha_x: torch.Tensor,
                 alpha_y: torch.Tensor) -> torch.Tensor:
        """
        Anisotropic GGX (Trowbridge-Reitz) NDF evaluated at a half-vector h.
        D(h) = 1 / (π α_x α_y  ((hx/αx)² + (hy/αy)² + hz²)²)
        The half-vector is already available in the lobe-local frame, so use
        its components directly instead of reconstructing hx/hy through
        sqrt(1 - cos(theta)^2). The direct form is equivalent and has a much
        better behaved gradient at normal incidence.
        """
        ax = alpha_x.clamp_min(1e-6)
        ay = alpha_y.clamp_min(1e-6)
        hx = m_local[..., 0]
        hy = m_local[..., 1]
        hz = m_local[..., 2].clamp_min(0.0)

        denom_sq = ((hx / ax) ** 2 + (hy / ay) ** 2 + hz ** 2) ** 2
        return 1.0 / (math.pi * ax * ay * denom_sq.clamp_min(1e-10))

    @staticmethod
    def _ggx_smith_g1(v_local: torch.Tensor,
                      alpha_x: torch.Tensor,
                      alpha_y: torch.Tensor) -> torch.Tensor:
        """
        Anisotropic GGX Smith G1 masking term.
        Uses the local direction components directly:
        G1(v) = 2 / (1 + sqrt(1 + ((alpha_x vx)^2 + (alpha_y vy)^2) / vz^2)).
        """
        ax = alpha_x.clamp_min(1e-6)
        ay = alpha_y.clamp_min(1e-6)
        vx = v_local[..., 0]
        vy = v_local[..., 1]
        vz = v_local[..., 2].clamp_min(1e-6)

        tan2_alpha2 = ((ax * vx) ** 2 + (ay * vy) ** 2) / (vz ** 2)
        return 2.0 / (1.0 + (1.0 + tan2_alpha2.clamp_min(0.0)).sqrt())

    # ------------------------------------------------------------------
    # Two-lobe GGX sampling (Heitz 2018 visible-normal sampling per lobe)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_ggx_vndf_single_lobe(
        wi_local: torch.Tensor,   # [B, 3] in lobe's local frame (z == N)
        alpha_x: torch.Tensor,    # [B]
        alpha_y: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """
        Sample a microfacet normal m from the GGX VNDF (Heitz 2018) in local
        space where the lobe normal is the z-axis.
        Returns m [B, 3] (unit, z >= 0 in the lobe frame).

        Algorithm from: Sampling the GGX Distribution of Visible Normals,
        Heitz 2018, JCGT.
        """
        device = wi_local.device
        dtype = wi_local.dtype
        B = wi_local.shape[0]

        # Step 1 — stretch wi by roughness
        wi_s = ImportanceSamplingDecoder._safe_normalize(
            torch.stack([
                wi_local[..., 0] * alpha_x,
                wi_local[..., 1] * alpha_y,
                wi_local[..., 2].clamp_min(1e-6),
            ], dim=-1)
        )

        # Step 2 — build orthonormal basis (t1, t2) around wi_s
        sign = torch.where(wi_s[..., 2] >= 0.0,
                           torch.ones(B, device=device, dtype=dtype),
                           -torch.ones(B, device=device, dtype=dtype))
        a = -1.0 / (sign + wi_s[..., 2]).clamp_min(1e-6)
        b_coeff = wi_s[..., 0] * wi_s[..., 1] * a
        t1 = torch.stack([
            1.0 + sign * wi_s[..., 0] ** 2 * a,
            sign * b_coeff,
            -sign * wi_s[..., 0],
        ], dim=-1)
        t2 = torch.stack([
            b_coeff,
            sign + wi_s[..., 1] ** 2 * a,
            -wi_s[..., 1],
        ], dim=-1)

        # Step 3 — sample point on disk parameterization
        u1 = torch.rand(B, device=device, dtype=dtype)
        u2 = torch.rand(B, device=device, dtype=dtype)

        r = u1.sqrt()
        phi = 2.0 * math.pi * u2
        t = r * torch.cos(phi)
        s = r * torch.sin(phi)
        t = t + s * (1.0 - t.abs()) * (
            torch.where(wi_s[..., 2] >= 0.0,
                        torch.zeros_like(s),
                        (1.0 - t.abs()) / (1.0 - wi_s[..., 2].abs()).clamp_min(1e-6))
        )

        # Step 4 — reproject onto hemisphere
        mh = (
            t.unsqueeze(-1) * t1
            + s.unsqueeze(-1) * t2
            + (1.0 - t**2 - s**2).clamp_min(0.0).sqrt().unsqueeze(-1) * wi_s
        )

        # Step 5 — un-stretch
        m = ImportanceSamplingDecoder._safe_normalize(
            torch.stack([
                mh[..., 0] * alpha_x,
                mh[..., 1] * alpha_y,
                mh[..., 2].clamp_min(0.0),
            ], dim=-1)
        )
        return m

    @staticmethod
    def _ggx_vndf_pdf(
        wi_local: torch.Tensor,   # [B, 3]
        m_local: torch.Tensor,    # [B, 3]  microfacet normal in lobe frame
        alpha_x: torch.Tensor,    # [B]
        alpha_y: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """
        PDF of the GGX VNDF: p(m) = G1(wi) D(m) |wi·m| / |wi·n|
        where n is the macro-surface normal (z-axis in local frame).
        Returns p(m) [B].  The reflection Jacobian |∂m/∂wo| = 1/(4|wo·m|)
        is applied by the caller to get p(wo).
        """
        cos_theta_i = wi_local[..., 2].clamp_min(1e-6)

        D = ImportanceSamplingDecoder._ggx_ndf(m_local, alpha_x, alpha_y)
        G1 = ImportanceSamplingDecoder._ggx_smith_g1(wi_local, alpha_x, alpha_y)
        wi_dot_m = (wi_local * m_local).sum(dim=-1).clamp_min(1e-6)
        return G1 * D * wi_dot_m / cos_theta_i.clamp_min(1e-6)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward_params(
        self, z: torch.Tensor, wi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict per-lobe GGX parameters from latent code z and incident direction wi.

        Returns:
            t, bv, n     : each [B, 2, 3]  — learned shading frames (one per lobe)
            alpha        : [B, 2, 2]       — (alpha_x, alpha_y) per lobe, in [ALPHA_MIN, 1]
            lobe_weights : [B, 2]          — mixture weights (sum to 1 via softmax)
        """
        t, bv, n = self._predict_frames(z)  # each [B, 2, 3]

        # Project wi into each lobe's frame
        wi_f = torch.stack([
            (wi.unsqueeze(1) * t).sum(dim=-1),
            (wi.unsqueeze(1) * bv).sum(dim=-1),
            (wi.unsqueeze(1) * n).sum(dim=-1),
        ], dim=-1)  # [B, 2, 3]

        dir_feats = wi_f.view(z.shape[0], 3 * self.num_frames)  # [B, 6]
        raw = self.mlp(torch.cat([z, dir_feats], dim=-1))       # [B, 10]
        raw = raw.view(z.shape[0], self.num_frames, 5)           # [B, 2, 5]

        # Slots 0,1: raw roughness -> alpha in [ALPHA_MIN, ALPHA_MAX]
        alpha = (
            self.ALPHA_MIN
            + (self.ALPHA_MAX - self.ALPHA_MIN) * torch.sigmoid(raw[..., 0:2])
        )  # [B, 2, 2]
        alpha = torch.nan_to_num(
            alpha,
            nan=0.5 * (self.ALPHA_MIN + self.ALPHA_MAX),
            posinf=self.ALPHA_MAX,
            neginf=self.ALPHA_MIN,
        ).clamp(self.ALPHA_MIN, self.ALPHA_MAX)

        # Slot 2: raw lobe weight -> softmax mixture
        lobe_weights = torch.softmax(raw[..., 2], dim=-1)  # [B, 2]
        lobe_weights = torch.nan_to_num(
            lobe_weights,
            nan=1.0 / float(self.num_frames),
            posinf=1.0,
            neginf=0.0,
        ).clamp_min(0.0)
        lobe_weights = lobe_weights / lobe_weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )

        return t, bv, n, alpha, lobe_weights

    def sample(
        self, z: torch.Tensor, wi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw one wo sample per batch element via the two-lobe GGX mixture.
        Returns:
            wo  : [B, 3]  sampled outgoing direction (world / shading space)
            pdf : [B]     probability density of wo under the mixture
        """
        t, bv, n, alpha, lobe_weights = self.forward_params(z, wi)
        B = z.shape[0]

        # --- Select which lobe to use for each sample (stratified by weight) ---
        lobe_idx = torch.multinomial(lobe_weights, num_samples=1).squeeze(-1)  # [B]

        # Gather the chosen lobe's frame and roughness
        idx = lobe_idx.view(B, 1, 1)
        t_sel  = t.gather(1,  idx.expand(B, 1, 3)).squeeze(1)   # [B, 3]
        bv_sel = bv.gather(1, idx.expand(B, 1, 3)).squeeze(1)
        n_sel  = n.gather(1,  idx.expand(B, 1, 3)).squeeze(1)
        alpha_sel = alpha.gather(1, lobe_idx.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)  # [B, 2]
        ax = alpha_sel[..., 0]
        ay = alpha_sel[..., 1]

        # Transform wi into the selected lobe's local frame
        wi_local = torch.stack([
            (wi * t_sel).sum(-1),
            (wi * bv_sel).sum(-1),
            (wi * n_sel).sum(-1),
        ], dim=-1)  # [B, 3]

        # Flip wi below the hemisphere to the upper side (back-facing case)
        wi_local = torch.where(
            wi_local[..., 2:3] < 0.0,
            wi_local * torch.tensor([1.0, 1.0, -1.0], device=wi.device, dtype=wi.dtype),
            wi_local,
        )

        # Sample microfacet normal m in local frame, then reflect
        m_local = self._sample_ggx_vndf_single_lobe(wi_local, ax, ay)      # [B, 3]
        wo_local = 2.0 * (wi_local * m_local).sum(-1, keepdim=True) * m_local - wi_local

        # Transform wo back to world / shading space
        wo = (
            wo_local[..., 0:1] * t_sel
            + wo_local[..., 1:2] * bv_sel
            + wo_local[..., 2:3] * n_sel
        )
        wo = self._safe_normalize(wo)

        # Evaluate the blended PDF at the sampled wo
        pdf = self.eval_pdf(z, wi, wo, _frames=(t, bv, n, alpha, lobe_weights))
        return wo, pdf

    def eval_pdf(
        self,
        z: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        _frames: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Evaluate the blended two-lobe GGX PDF p(wo | wi, z).
        p(wo) = sum_i  weight_i * p_i(wo)
        where p_i(wo) = p_vndf_i(m_i) / (4 |wo · m_i|)  and  m_i = normalize(wi + wo).
        """
        if _frames is None:
            t, bv, n, alpha, lobe_weights = self.forward_params(z, wi)
        else:
            t, bv, n, alpha, lobe_weights = _frames

        B = z.shape[0]
        pdf_total = torch.zeros(B, device=z.device, dtype=z.dtype)

        for i in range(self.num_frames):
            t_i  = t[:, i, :]   # [B, 3]
            bv_i = bv[:, i, :]
            n_i  = n[:, i, :]
            ax_i = alpha[:, i, 0]
            ay_i = alpha[:, i, 1]
            w_i  = lobe_weights[:, i]  # [B]

            # wi and wo in lobe i's local frame
            wi_local = torch.stack([
                (wi * t_i).sum(-1),
                (wi * bv_i).sum(-1),
                (wi * n_i).sum(-1),
            ], dim=-1)
            wo_local = torch.stack([
                (wo * t_i).sum(-1),
                (wo * bv_i).sum(-1),
                (wo * n_i).sum(-1),
            ], dim=-1)

            # Flip below-hemisphere wi (same as in sample())
            wi_local = torch.where(
                wi_local[..., 2:3] < 0.0,
                wi_local * torch.tensor([1.0, 1.0, -1.0], device=wi.device, dtype=wi.dtype),
                wi_local,
            )

            # Half-vector m = normalize(wi + wo) in local frame
            m_local = self._safe_normalize(wi_local + wo_local)
            # Ensure m is on the upper hemisphere
            m_local = torch.where(
                m_local[..., 2:3] < 0.0, -m_local, m_local
            )

            wo_dot_m = (wo_local * m_local).sum(-1).abs().clamp_min(1e-6)

            p_m = self._ggx_vndf_pdf(wi_local, m_local, ax_i, ay_i)  # p(m)
            p_wo = p_m / (4.0 * wo_dot_m)                            # p(wo)

            p_wo = torch.nan_to_num(p_wo, nan=0.0, posinf=0.0, neginf=0.0)
            w_i = torch.nan_to_num(w_i, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)

            pdf_total = pdf_total + w_i * p_wo.clamp_min(0.0)

        return torch.nan_to_num(pdf_total, nan=1e-10, posinf=1e10, neginf=1e-10).clamp_min(1e-10)


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


class NeuralMaterialModel(nn.Module):
    """
    Wraps:
      - LatentTexture
            - Decoder (BRDF evaluation)
            - ImportanceSamplingDecoder (direction sampling)
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.latent = LatentTexture(cfg.tex_h, cfg.tex_w, cfg.latent_ch)
        self.decoder = Decoder(
            latent_ch=cfg.latent_ch,
            num_frames=cfg.num_frames,
            mlp_width=cfg.mlp_width,
            mlp_depth=cfg.mlp_depth,
            use_bias_in_mlp=cfg.use_bias_in_mlp,
            frame_linear_bias=cfg.frame_linear_bias,
            exp_offset=cfg.exp_offset,
        )
        self.importance_sampler = ImportanceSamplingDecoder(
            latent_ch=cfg.latent_ch,
            num_frames=cfg.num_frames,
            mlp_width=cfg.sampler_mlp_width,
            mlp_depth=cfg.sampler_mlp_depth,
            use_bias_in_mlp=cfg.use_bias_in_mlp,
            frame_linear_bias=cfg.frame_linear_bias,
        )
        self.encoder = MaterialEncoder(
            input_ch=get_encoder_input_dim(cfg),
            latent_ch=cfg.latent_ch,
            hidden_width=cfg.encoder_width,
            depth=cfg.encoder_depth,
        )

    def decode_with_raw(
        self, z: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.decoder.forward_raw(z, wi, wo)
        y_hat = torch.exp(raw - self.decoder.exp_offset)
        return y_hat, raw

    def forward_with_raw(
        self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.latent.sample(uv)
        return self.decode_with_raw(z, wi, wo)

    def forward(
        self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
    ) -> torch.Tensor:
        y_hat, _raw = self.forward_with_raw(uv, wi, wo)
        return y_hat

    def sample_directions(
        self, uv: torch.Tensor, wi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.latent.sample(uv)
        return self.importance_sampler.sample(z, wi)

    def eval_sampling_pdf(
        self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
    ) -> torch.Tensor:
        z = self.latent.sample(uv)
        return self.importance_sampler.eval_pdf(z, wi, wo)

def to_local(
    v_world: torch.Tensor, t: torch.Tensor, b: torch.Tensor, n: torch.Tensor
) -> torch.Tensor:
    """
    World->local using basis (t,b,n) as rows via dot products.

    v_world: [B,3]
    returns v_local: [B,3] such that local z aligns with n.
    """
    return torch.stack(
        [
            (v_world * t).sum(dim=1),
            (v_world * b).sum(dim=1),
            (v_world * n).sum(dim=1),
        ],
        dim=1,
    )


# =============================================================================
# Loss / Metrics
# =============================================================================


def log_l1_loss(
    y_hat: torch.Tensor, y: torch.Tensor, eps: float, mask_threshold: float = 1e-4
) -> torch.Tensor:
    """
    L1 loss in log space:
      mean(|log(y_hat+eps) - log(y+eps)|)
    """
    # Build per-sample mask: keep samples that have at least one significant channel
    valid = y.amax(dim=-1) >= mask_threshold  # [B]
    if valid.any():
        y_hat_c = y_hat[valid].clamp_min(eps)
        y_c = y[valid].clamp_min(eps)
    else:
        # Fallback: use everything (avoids zero-element mean on pathological batches)
        y_hat_c = y_hat.clamp_min(eps)
        y_c = y.clamp_min(eps)

    return (torch.log(y_hat_c) - torch.log(y_c)).abs().mean()


@torch.no_grad()
def compute_basic_stats(y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    mae = (y_hat - y).abs().mean().item()
    rel = ((y_hat - y).abs() / y.clamp_min(1e-6)).mean().item()
    yh_mean = y_hat.mean().item()
    y_mean = y.mean().item()
    return {"mae": mae, "rel_mae": rel, "yhat_mean": yh_mean, "y_mean": y_mean}


@torch.no_grad()
def compute_raw_stats(raw: torch.Tensor) -> Dict[str, float]:
    return {
        "raw_mean": raw.mean().item(),
        "raw_std": raw.std(unbiased=False).item(),
        "raw_min": raw.min().item(),
        "raw_max": raw.max().item(),
    }


def importance_sampling_loss(
    model: NeuralMaterialModel,
    z: torch.Tensor,
    wi: torch.Tensor,
    wo: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Trains sampler by maximizing probability of directions weighted by BRDF magnitude.
    """
    y_true = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
    pdf_sampler = model.importance_sampler.eval_pdf(z, wi, wo)
    pdf_sampler = torch.nan_to_num(pdf_sampler, nan=eps, posinf=1e6, neginf=eps).clamp(
        min=eps, max=1e6
    )

    brdf_weights = (y_true.sum(dim=-1) / (3.0 + eps)).clamp_min(0.0)
    loss_terms = -torch.log(pdf_sampler) * brdf_weights
    finite_mask = torch.isfinite(loss_terms)

    if not finite_mask.any():
        # Keep graph connected while avoiding a hard failure on a pathological batch.
        return (pdf_sampler * 0.0).sum()

    return loss_terms[finite_mask].mean()


def build_material_features(
    batch: Dict[str, torch.Tensor], cfg: TrainConfig, device: torch.device
) -> torch.Tensor:
    features = []

    if cfg.use_albedo_features:
        features.append(batch["albedo"].to(device, non_blocking=True))

    if cfg.use_spec_features:
        features.append(batch["spec"].to(device, non_blocking=True))

    if cfg.use_normal_features:
        normal = F.normalize(
            batch["normal"].to(device, non_blocking=True), dim=-1, eps=1e-8
        )
        features.append(normal)

    if cfg.use_roughness_feature:
        roughness = batch["roughness"].to(device, non_blocking=True).unsqueeze(-1)
        features.append(roughness)

    if cfg.use_pdf_feature:
        pdf = (
            torch.log1p(batch["pdf"].to(device, non_blocking=True).clamp_min(0.0))
            .unsqueeze(-1)
        )
        features.append(pdf)

    if not features:
        raise ValueError("Encoder feature set is empty. Enable at least one training feature.")

    return torch.cat(features, dim=-1)


def get_training_phase(cfg: TrainConfig, epoch: int) -> str:
    if epoch < cfg.encoder_bootstrap_epochs:
        return "bootstrap"
    return "finetune"


@torch.no_grad()
def initialize_latent_texture_from_encoder(
    model: NeuralMaterialModel, cfg: TrainConfig, generation_index: int
) -> None:
    if get_encoder_input_dim(cfg) == 0:
        raise ValueError("Cannot initialize latent texture without encoder features.")

    device = torch.device(cfg.device)
    model.eval()

    print(
        f"[bootstrap] Initializing latent texture from encoder on a full "
        f"{cfg.tex_w}x{cfg.tex_h} UV grid in a single batch"
    )
    sample_count = cfg.tex_w * cfg.tex_h
    grid_generator = DataGenerator(sampleCount=sample_count)
    try:
        grid_batch = grid_generator.generate_grid_data(
            cfg.tex_w,
            cfg.tex_h,
            cfg.seed,
            SEED_DOMAIN_BOOTSTRAP,
            generation_index,
        ).copy()
    finally:
        grid_generator.release_data()

    grid_tensor = tensorize_batch(data_to_dict(grid_batch))
    latent_chunks = []

    for start in range(0, sample_count, cfg.latent_init_batch_size):
        end = min(start + cfg.latent_init_batch_size, sample_count)
        chunk = {key: value[start:end] for key, value in grid_tensor.items()}
        features = build_material_features(chunk, cfg, device)
        latent_chunks.append(model.encoder(features).cpu())

    latent_image = torch.cat(latent_chunks, dim=0).view(
        cfg.tex_h, cfg.tex_w, cfg.latent_ch
    )

    z_image = latent_image.permute(2, 0, 1).unsqueeze(0).contiguous()
    model.latent.Z.copy_(z_image.to(device))


# =============================================================================
# Training utilities
# =============================================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _latent_lr(cfg: TrainConfig) -> float:
    return cfg.lr if cfg.lr_latent is None else cfg.lr_latent


def _decoder_lr(cfg: TrainConfig) -> float:
    return cfg.lr if cfg.lr_decoder is None else cfg.lr_decoder


def _latent_lr_min(cfg: TrainConfig) -> float:
    return cfg.lr_min if cfg.lr_latent is None else min(cfg.lr_min, _latent_lr(cfg))


def _decoder_lr_min(cfg: TrainConfig) -> float:
    return cfg.lr_min if cfg.lr_decoder is None else min(cfg.lr_min, _decoder_lr(cfg))


def make_optimizer(
    model: NeuralMaterialModel, cfg: TrainConfig, phase: str
) -> torch.optim.Optimizer:
    param_groups = []
    if phase == "finetune" and cfg.train_latent_texture:
        latent_params = [p for p in model.latent.parameters() if p.requires_grad]
        if latent_params:
            param_groups.append(
                {"params": latent_params, "lr": _latent_lr(cfg), "name": "latent"}
            )
    if cfg.train_decoder:
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        if decoder_params:
            param_groups.append(
                {"params": decoder_params, "lr": _decoder_lr(cfg), "name": "decoder"}
            )
        if phase == "bootstrap":
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append(
                    {"params": encoder_params, "lr": _decoder_lr(cfg), "name": "encoder"}
                )
    if cfg.train_importance_sampler:
        sampler_params = [p for p in model.importance_sampler.parameters() if p.requires_grad]
        if sampler_params:
            param_groups.append(
                {"params": sampler_params, "lr": _decoder_lr(cfg), "name": "sampler"}
            )
    if not param_groups:
        raise ValueError(
            f"Nothing to train during {phase}: active parameter groups are empty"
        )
    return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)


def make_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    """
    Cosine LR decay from the per-group base LR to the per-group minimum over cfg.max_epochs (epoch-stepped).
    """
    base_by_name = {
        "latent": _latent_lr(cfg),
        "decoder": _decoder_lr(cfg),
        "encoder": _decoder_lr(cfg),
        "sampler": _decoder_lr(cfg),
    }
    min_by_name = {
        "latent": _latent_lr_min(cfg),
        "decoder": _decoder_lr_min(cfg),
        "encoder": _decoder_lr_min(cfg),
        "sampler": _decoder_lr_min(cfg),
    }

    def lr_lambda_factory(group_name: str):
        base = base_by_name.get(group_name, base_by_name["decoder"])
        min_lr = min_by_name.get(group_name, min_by_name["decoder"])

        def lr_lambda(epoch: int):
            if cfg.max_epochs <= 1:
                return min_lr / max(base, 1e-12)
            t = min(epoch / max(cfg.max_epochs - 1, 1), 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))
            lr_now = min_lr + (base - min_lr) * scale
            return lr_now / max(base, 1e-12)

        return lr_lambda

    lambdas = [lr_lambda_factory(pg.get("name", "latent")) for pg in opt.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(opt, lambdas)


def maybe_freeze_parts(
    model: NeuralMaterialModel,
    cfg: TrainConfig,
    *,
    epoch: Optional[int] = None,
) -> None:
    if epoch is not None:
        if (
            cfg.freeze_latent_after_epoch is not None
            and epoch >= cfg.freeze_latent_after_epoch
        ):
            for p in model.latent.parameters():
                p.requires_grad_(False)
        if (
            cfg.freeze_decoder_after_epoch is not None
            and epoch >= cfg.freeze_decoder_after_epoch
        ):
            for module in (model.decoder, model.encoder):
                for p in module.parameters():
                    p.requires_grad_(False)


def maybe_rebuild_optimizer_and_scheduler(
    model: NeuralMaterialModel,
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    phase: str,
):
    active_group_names = []
    if phase == "finetune" and cfg.train_latent_texture and any(
        p.requires_grad for p in model.latent.parameters()
    ):
        active_group_names.append("latent")
    if cfg.train_decoder and any(p.requires_grad for p in model.decoder.parameters()):
        active_group_names.append("decoder")
    if (
        phase == "bootstrap"
        and cfg.train_decoder
        and any(p.requires_grad for p in model.encoder.parameters())
    ):
        active_group_names.append("encoder")
    if cfg.train_importance_sampler and any(p.requires_grad for p in model.importance_sampler.parameters()):
        active_group_names.append("sampler")

    current_group_names = [pg.get("name") for pg in opt.param_groups]
    if active_group_names == current_group_names:
        return opt, scheduler

    # Build a mapping from parameter data_ptr -> old state so we can transfer moments
    old_state = opt.state  # dict keyed by parameter tensor

    new_opt = make_optimizer(model, cfg, phase)

    for pg in new_opt.param_groups:
        for p in pg["params"]:
            if p in old_state and len(old_state[p]) > 0:
                new_opt.state[p] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in old_state[p].items()
                }

    new_scheduler = make_scheduler(new_opt, cfg)
    if scheduler is not None and hasattr(scheduler, "last_epoch"):
        new_scheduler.last_epoch = scheduler.last_epoch

    print(
        f"[train] rebuilt optimizer groups: {current_group_names} -> {active_group_names}"
    )
    return new_opt, new_scheduler


def _clear_param_group_grads_except(
    opt: torch.optim.Optimizer, keep_group_names: set[str]
) -> None:
    for pg in opt.param_groups:
        if pg.get("name") in keep_group_names:
            continue
        for p in pg["params"]:
            p.grad = None


def _maybe_transform_dirs_with_normals(
    batch: Dict[str, torch.Tensor], cfg: TrainConfig, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direction transforms are disabled. Sampled guide normals are treated as
    training-side material parameters only, so wi/wo are always returned as-is.
    """
    wi = batch["wi"].to(device, non_blocking=True)
    wo = batch["wo"].to(device, non_blocking=True)
    return wi, wo

def train_one_epoch(
    model: NeuralMaterialModel,
    batch_decoder: Dict[str, torch.Tensor],
    batch_sampler: Optional[Dict[str, torch.Tensor]],
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    epoch: int,
    phase: str,
):
    model.train()
    device = torch.device(cfg.device)
    decoder_frozen_logged = False
    latent_frozen_logged = False

    opt, scheduler = maybe_rebuild_optimizer_and_scheduler(
        model, opt, scheduler, cfg, phase
    )

    decoder_now_frozen = all(not p.requires_grad for p in model.decoder.parameters())
    if decoder_now_frozen and not decoder_frozen_logged:
        print(f"[train] freezing decoder at epoch={epoch}")
        decoder_frozen_logged = True

    latent_now_frozen = all(not p.requires_grad for p in model.latent.parameters())
    if latent_now_frozen and not latent_frozen_logged:
        print(f"[train] freezing latent texture at epoch={epoch}")
        latent_frozen_logged = True

    uv_dec = batch_decoder["uv"].to(device, non_blocking=True)
    y_dec = batch_decoder["y"].to(device, non_blocking=True)

    wi_dec, wo_dec = _maybe_transform_dirs_with_normals(batch_decoder, cfg, device)
    if cfg.clamp_min_target > 0.0:
        y_dec = y_dec.clamp_min(cfg.clamp_min_target)

    if phase == "bootstrap":
        material_features_dec = build_material_features(batch_decoder, cfg, device)
        z_dec = model.encoder(material_features_dec)
    else:
        z_dec = model.latent.sample(uv_dec)

    y_hat_dec, raw_dec = model.decode_with_raw(z_dec, wi_dec, wo_dec)
    bsdf_loss = log_l1_loss(y_hat_dec, y_dec, cfg.log_eps)

    if not torch.isfinite(bsdf_loss):
        raise RuntimeError(f"Non-finite BRDF loss at epoch {epoch}: {bsdf_loss.item()}")

    opt.zero_grad(set_to_none=True)
    bsdf_loss.backward()
    _clear_param_group_grads_except(opt, {"latent", "decoder", "encoder"})

    if cfg.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], cfg.grad_clip_norm
        )

    opt.step()

    sampler_loss = None

    if cfg.train_importance_sampler:
        sampler_batch = batch_decoder if batch_sampler is None else batch_sampler

        uv_sam = sampler_batch["uv"].to(device, non_blocking=True)
        y_sam = sampler_batch["y"].to(device, non_blocking=True)
        wi_sam, wo_sam = _maybe_transform_dirs_with_normals(sampler_batch, cfg, device)

        if cfg.clamp_min_target > 0.0:
            y_sam = y_sam.clamp_min(cfg.clamp_min_target)

        if phase == "bootstrap":
            material_features_sam = build_material_features(sampler_batch, cfg, device)
            z_sam = model.encoder(material_features_sam).detach()
        else:
            z_sam = model.latent.sample(uv_sam).detach()

        sampler_loss = importance_sampling_loss(model, z_sam, wi_sam, wo_sam, y_sam, cfg.log_eps)
        if not torch.isfinite(sampler_loss):
            raise RuntimeError(f"Non-finite sampler loss at epoch {epoch}: {sampler_loss.item()}")

        opt.zero_grad(set_to_none=True)
        sampler_loss.backward()
        _clear_param_group_grads_except(opt, {"sampler"})

        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                [p for p in model.importance_sampler.parameters() if p.grad is not None],
                cfg.grad_clip_norm,
            )

        opt.step()

    with torch.no_grad():
        stats = compute_basic_stats(y_hat_dec, y_dec)
        raw_stats = compute_raw_stats(raw_dec)

    out = {
        "brdf_loss": bsdf_loss.item(),
        "phase": phase,
        "mae": stats["mae"],
        "yhat_mean": stats["yhat_mean"],
        "y_mean": stats["y_mean"],
        "raw_mean": raw_stats["raw_mean"],
        "raw_std": raw_stats["raw_std"],
    }
    if sampler_loss is not None:
        out["sampler_loss"] = sampler_loss.item()

    return (out, opt, scheduler)


@torch.no_grad()
def validate(
    model: NeuralMaterialModel,
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    epoch: int,
    phase: str,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)

    uv = batch["uv"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)

    wi, wo = _maybe_transform_dirs_with_normals(batch, cfg, device)
    if cfg.clamp_min_target > 0.0:
        y = y.clamp_min(cfg.clamp_min_target)

    if phase == "bootstrap":
        material_features = build_material_features(batch, cfg, device)
        z = model.encoder(material_features)
    else:
        z = model.latent.sample(uv)

    y_hat, raw = model.decode_with_raw(z, wi, wo)
    loss = log_l1_loss(y_hat, y, cfg.log_eps)
    stats = compute_basic_stats(y_hat, y)
    raw_stats = compute_raw_stats(raw)

    out = {
        "phase": phase,
        "val_loss": loss.item(),
        "val_mae": stats["mae"],
        "val_yhat_mean": stats["yhat_mean"],
        "val_y_mean": stats["y_mean"],
        "val_raw_mean": raw_stats["raw_mean"],
        "val_raw_std": raw_stats["raw_std"],
    }
    return out


# =============================================================================
# Export / Checkpoints
# =============================================================================


def snapshot_model_state(model: NeuralMaterialModel) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def save_checkpoint(
    model: NeuralMaterialModel,
    opt,
    scheduler,
    cfg: TrainConfig,
    epoch: int,
    metrics: Dict[str, float],
    filename: str = "checkpoint_epoch.pt",
) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, filename)
    payload = {
        "epoch": epoch,
        "config": asdict(cfg),
        "metrics": metrics,
        "model": model.state_dict(),
        "optimizer": None if opt is None else opt.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_model_weights_from_checkpoint(
    model: NeuralMaterialModel, ckpt_path: str, device: torch.device
) -> Dict:
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    return payload


def export_latent_texture(model: NeuralMaterialModel, cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    Z = model.latent.Z.detach().cpu()  # [1,C,H,W]
    torch.save(
        {"Z": Z, "shape": (cfg.tex_h, cfg.tex_w, cfg.latent_ch)},
        os.path.join(cfg.out_dir, "latent_texture.pt"),
    )

    Z_np = Z.numpy()
    if cfg.latent_ch == 8:
        rgba0 = Z_np[0, 0:4, :, :]  # [4,H,W]
        rgba1 = Z_np[0, 4:8, :, :]
        np.savez_compressed(os.path.join(cfg.out_dir, "latent_rgba0.npz"), rgba=rgba0)
        np.savez_compressed(os.path.join(cfg.out_dir, "latent_rgba1.npz"), rgba=rgba1)
    else:
        np.savez_compressed(os.path.join(cfg.out_dir, "latent_all.npz"), z=Z_np)


def export_decoder_weights(model: NeuralMaterialModel, cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    torch.save(model.decoder.state_dict(), os.path.join(cfg.out_dir, "decoder.pt"))

    sd = model.decoder.state_dict()
    out = {
        "latent_ch": np.array([cfg.latent_ch], dtype=np.int32),
        "num_frames": np.array([cfg.num_frames], dtype=np.int32),
    }

    out["frame_linear.weight"] = sd["frame_linear.weight"].detach().cpu().numpy()
    if "frame_linear.bias" in sd:
        out["frame_linear.bias"] = sd["frame_linear.bias"].detach().cpu().numpy()

    for k, v in sd.items():
        if k.startswith("mlp."):
            out[k] = v.detach().cpu().numpy()

    np.savez_compressed(os.path.join(cfg.out_dir, "decoder_weights.npz"), **out)


def write_exr(path: Path, rgba_hw4: np.ndarray) -> None:
    rgba_hw4 = np.asarray(rgba_hw4, dtype=np.float32)
    assert rgba_hw4.ndim == 3 and rgba_hw4.shape[2] == 4, f"Expected HxWx4, got {rgba_hw4.shape}"

    h, w, _ = rgba_hw4.shape

    try:
        import OpenEXR
        import Imath

        header = OpenEXR.Header(w, h)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        header["channels"] = {
            "R": Imath.Channel(pt),
            "G": Imath.Channel(pt),
            "B": Imath.Channel(pt),
            "A": Imath.Channel(pt),
        }

        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels(
            {
                "R": rgba_hw4[:, :, 0].astype(np.float32).tobytes(),
                "G": rgba_hw4[:, :, 1].astype(np.float32).tobytes(),
                "B": rgba_hw4[:, :, 2].astype(np.float32).tobytes(),
                "A": rgba_hw4[:, :, 3].astype(np.float32).tobytes(),
            }
        )
        exr.close()
        return
    except Exception:
        pass


def save_weights_bin(path: Path, weights: dict) -> None:
    return save_network_weights_bin(
        path=path,
        weights=weights,
        input_dim=20,
        output_dim=3,
    )


def _infer_network_layout(weights: dict, *, input_dim: int, output_dim: int) -> dict:
    frame_weight = np.asarray(weights["frame_linear.weight"], dtype=np.float32)
    if frame_weight.shape != (12, 8):
        raise ValueError(f"frame_linear.weight expected shape (12, 8), got {frame_weight.shape}")

    latent_ch = int(np.asarray(weights.get("latent_ch", np.array([8], dtype=np.int32))).reshape(-1)[0])
    num_frames = int(np.asarray(weights.get("num_frames", np.array([2], dtype=np.int32))).reshape(-1)[0])

    layer_indices = sorted(
        int(k.split(".")[1]) for k in weights.keys() if k.startswith("mlp.") and k.endswith(".weight")
    )
    if not layer_indices:
        raise ValueError("No MLP layers found in weights.")

    linear_layers = ["frame_linear"] + [f"mlp.{idx}" for idx in layer_indices]
    mlp_depth = len(layer_indices) - 1
    if mlp_depth not in (2, 3):
        raise ValueError(f"Unsupported MLP depth: {mlp_depth}. Expected 2 or 3.")

    first_w = np.asarray(weights[f"mlp.{layer_indices[0]}.weight"], dtype=np.float32)
    mlp_width = int(first_w.shape[0])

    expected_prev = input_dim
    decoder_layout = {"frame_linear.weight": [12, 8]}
    for idx in layer_indices:
        w_name = f"mlp.{idx}.weight"
        b_name = f"mlp.{idx}.bias"
        w = np.asarray(weights[w_name], dtype=np.float32)
        b = np.asarray(weights[b_name], dtype=np.float32)

        out_dim = output_dim if idx == layer_indices[-1] else mlp_width
        if tuple(w.shape) != (out_dim, expected_prev):
            raise ValueError(f"{w_name} expected shape {(out_dim, expected_prev)}, got {w.shape}")
        if tuple(b.shape) != (out_dim,):
            raise ValueError(f"{b_name} expected shape {(out_dim,)}, got {b.shape}")

        decoder_layout[w_name] = [int(w.shape[0]), int(w.shape[1])]
        decoder_layout[b_name] = [int(b.shape[0])]
        expected_prev = mlp_width

    if latent_ch != 8:
        raise ValueError(f"Only latent_ch=8 is supported for runtime export, got {latent_ch}")
    if num_frames != 2:
        raise ValueError(f"Only num_frames=2 is supported for runtime export, got {num_frames}")
    if mlp_width not in (16, 32, 64):
        raise ValueError(f"Unsupported mlp_width={mlp_width}. Expected one of 16, 32, 64.")

    return {
        "latent_ch": latent_ch,
        "num_frames": num_frames,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "linear_layers": linear_layers,
        "decoder_layout": decoder_layout,
    }


def save_network_weights_bin(
    path: Path,
    weights: dict,
    *,
    input_dim: int,
    output_dim: int,
) -> dict:
    layout = _infer_network_layout(weights, input_dim=input_dim, output_dim=output_dim)
    latent_ch = layout["latent_ch"]
    num_frames = layout["num_frames"]
    mlp_width = layout["mlp_width"]
    mlp_depth = layout["mlp_depth"]

    with open(path, "wb") as f:
        f.write(b"NMDLWT02")
        f.write(struct.pack("<iiii", latent_ch, num_frames, mlp_width, mlp_depth))

        for layer_name in layout["linear_layers"]:
            weight_name = f"{layer_name}.weight"
            bias_name = f"{layer_name}.bias"
            w = np.asarray(weights[weight_name], dtype=np.float32)
            f.write(w.ravel(order="C").tobytes())
            if bias_name in weights:
                b = np.asarray(weights[bias_name], dtype=np.float32)
                f.write(b.ravel(order="C").tobytes())

    layout["weight_file_format"] = "NMDLWT02"
    return layout


def save_metadata(path: Path, latent: np.ndarray, weights: dict) -> None:
    _, h, w = latent.shape
    layout = _infer_network_layout(weights, input_dim=20, output_dim=3)
    metadata = {
        "width": int(w),
        "height": int(h),
        "latent_dim": int(latent.shape[0]),
        "num_frames": layout["num_frames"],
        "apply_exp": True,
        "mlp_width": layout["mlp_width"],
        "mlp_depth": layout["mlp_depth"],
        "weight_file_format": "NMDLWT02",
        "decoder_layout": layout["decoder_layout"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_sampler_metadata(path: Path, weights: dict) -> None:
    # Sampler head outputs 5 values per lobe × 2 lobes = 10 channels.
    layout = _infer_network_layout(weights, input_dim=14, output_dim=10)
    metadata = {
        "latent_dim": layout["latent_ch"],
        "num_frames": layout["num_frames"],
        "mlp_width": layout["mlp_width"],
        "mlp_depth": layout["mlp_depth"],
        "weight_file_format": "NMDLWT02",
        "decoder_layout": layout["decoder_layout"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def export_renderer_assets(model: NeuralMaterialModel, cfg: TrainConfig) -> None:
    preview_dir = Path(cfg.preview_out_dir) if cfg.preview_out_dir else Path(__file__).resolve().parents[2] / "MatXScenes" / "Preview"
    os.makedirs(preview_dir, exist_ok=True)

    if cfg.latent_ch != 8:
        print(
            f"[export] Skipping renderer-ready latent/weight bundle because latent_ch={cfg.latent_ch}; expected 8."
        )
        return

    latent = model.latent.Z.detach().cpu().numpy()[0]
    rgba0 = latent[0:4].transpose(1, 2, 0).copy()
    rgba1 = latent[4:8].transpose(1, 2, 0).copy()

    write_exr(preview_dir / "latent0.exr", rgba0)
    write_exr(preview_dir / "latent1.exr", rgba1)

    np.save(preview_dir / "latent0.npy", rgba0)
    np.save(preview_dir / "latent1.npy", rgba1)

    sd = model.decoder.state_dict()
    weights = {
        "latent_ch": np.array([cfg.latent_ch], dtype=np.int32),
        "num_frames": np.array([cfg.num_frames], dtype=np.int32),
        "frame_linear.weight": sd["frame_linear.weight"].detach().cpu().numpy(),
    }
    for key, value in sd.items():
        if key.startswith("mlp."):
            weights[key] = value.detach().cpu().numpy()

    save_weights_bin(preview_dir / "decoder_weights.bin", weights)
    save_metadata(preview_dir / "metadata.json", latent, weights)

    if cfg.train_importance_sampler:
        sampler_sd = model.importance_sampler.state_dict()
        sampler_weights = {
            "latent_ch": np.array([cfg.latent_ch], dtype=np.int32),
            "num_frames": np.array([cfg.num_frames], dtype=np.int32),
            "frame_linear.weight": sampler_sd["frame_linear.weight"].detach().cpu().numpy(),
        }
        for key, value in sampler_sd.items():
            if key.startswith("mlp."):
                sampler_weights[key] = value.detach().cpu().numpy()

        save_network_weights_bin(
            preview_dir / "sampler_weights.bin",
            sampler_weights,
            input_dim=14,
            output_dim=10,
        )
        save_sampler_metadata(preview_dir / "sampler_metadata.json", sampler_weights)

    print(f"[export] Renderer-ready assets written to: {preview_dir}")


def save_config(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()

    p.add_argument("--out_dir", type=str, default="./output_weights")
    p.add_argument(
        "--preview_out_dir",
        type=str,
        default="",
        help="Directory for final renderer-ready assets. Defaults to MatXScenes/Preview.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--tex_h", type=int, default=4096)
    p.add_argument("--tex_w", type=int, default=4096)
    p.add_argument("--latent_ch", type=int, default=8)

    p.add_argument("--num_frames", type=int, default=2)
    p.add_argument("--mlp_width", type=int, default=32)
    p.add_argument("--mlp_depth", type=int, default=2)
    p.add_argument("--sampler_mlp_width", type=int, default=32)
    p.add_argument("--sampler_mlp_depth", type=int, default=2)
    p.add_argument("--exp_offset", type=float, default=3.0)

    p.add_argument("--training_n", type=int, default=65536)
    p.add_argument("--validation_size", type=int, default=65536)
    p.add_argument("--max_epochs", type=int, default=300000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--lr_latent", type=float, default=None)
    p.add_argument("--lr_decoder", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=None)

    p.add_argument("--log_eps", type=float, default=1e-6)
    p.add_argument("--clamp_min_target", type=float, default=0.0)

    p.add_argument("--print_every_epochs", type=int, default=10000)
    p.add_argument(
        "--train_latent_texture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable latent texture training.",
    )
    p.add_argument(
        "--train_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable decoder training.",
    )
    p.add_argument(
        "--train_importance_sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable training of the importance sampling decoder.",
    )

    p.add_argument("--freeze_latent_after_epoch", type=int, default=None)
    p.add_argument("--freeze_decoder_after_epoch", type=int, default=None)

    p.add_argument(
        "--use_normals",
        action="store_true",
        help="Legacy no-op kept for CLI compatibility. Sampled guide normals stay on the training/material side only.",
    )
    p.add_argument("--encoder_width", type=int, default=32)
    p.add_argument("--encoder_depth", type=int, default=2)
    p.add_argument(
        "--encoder_bootstrap_epochs",
        type=int,
        default=2000,
        help="Number of epochs to train encoder -> decoder directly before initializing the latent texture.",
    )
    p.add_argument(
        "--latent_init_batch_size",
        type=int,
        default=65536,
        help="Batch size used when initializing the latent texture from encoder outputs.",
    )
    p.add_argument(
        "--albedo_feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable albedo in the training-only material encoder.",
    )
    p.add_argument(
        "--spec_feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable specular in the training-only material encoder.",
    )
    p.add_argument(
        "--normal_feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable guide normal in the training-only material encoder.",
    )
    p.add_argument(
        "--roughness_feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable roughness in the training-only material encoder.",
    )
    p.add_argument(
        "--pdf_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable BSDF pdf in the training-only material encoder.",
    )

    args = p.parse_args()

    cfg = TrainConfig()
    cfg.out_dir = args.out_dir
    cfg.preview_out_dir = args.preview_out_dir
    cfg.device = args.device
    cfg.seed = args.seed

    cfg.tex_h = args.tex_h
    cfg.tex_w = args.tex_w
    cfg.latent_ch = args.latent_ch

    cfg.num_frames = args.num_frames
    cfg.mlp_width = args.mlp_width
    cfg.mlp_depth = args.mlp_depth
    cfg.sampler_mlp_width = args.sampler_mlp_width
    cfg.sampler_mlp_depth = args.sampler_mlp_depth
    cfg.exp_offset = args.exp_offset

    cfg.training_n = args.training_n
    cfg.validation_n = args.validation_size
    cfg.max_epochs = args.max_epochs
    cfg.lr = args.lr
    cfg.lr_min = args.lr_min
    cfg.lr_latent = args.lr_latent
    cfg.lr_decoder = args.lr_decoder
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip_norm = args.grad_clip_norm

    cfg.log_eps = args.log_eps
    cfg.clamp_min_target = args.clamp_min_target

    cfg.print_every_epochs = max(0, args.print_every_epochs)
    cfg.train_latent_texture = args.train_latent_texture
    cfg.train_decoder = args.train_decoder

    cfg.freeze_latent_after_epoch = args.freeze_latent_after_epoch
    cfg.freeze_decoder_after_epoch = args.freeze_decoder_after_epoch

    cfg.use_normals = args.use_normals
    cfg.encoder_width = args.encoder_width
    cfg.encoder_depth = args.encoder_depth
    cfg.encoder_bootstrap_epochs = max(0, args.encoder_bootstrap_epochs)
    cfg.latent_init_batch_size = max(1, args.latent_init_batch_size)
    cfg.use_albedo_features = args.albedo_feature
    cfg.use_spec_features = args.spec_feature
    cfg.use_normal_features = args.normal_feature
    cfg.use_roughness_feature = args.roughness_feature
    cfg.use_pdf_feature = args.pdf_feature

    cfg.train_importance_sampler = args.train_importance_sampler

    return cfg


def data_to_dict(data: np.ndarray):
    uv = data[:, 0:2]
    wo = data[:, 2:5]
    wi = data[:, 5:8]
    f = data[:, 8:11]
    spec = data[:, 11:14]
    albedo = data[:, 14:17]
    normal = data[:, 17:20]
    roughness = data[:, 20]
    pdf = data[:, 21]

    return {
        "uv": uv,
        "wo": wo,
        "wi": wi,
        "y": f,
        "spec": spec,
        "albedo": albedo,
        "normal": normal,
        "roughness": roughness,
        "pdf": pdf,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    run_start_time = time.time()

    if cfg.encoder_bootstrap_epochs > 0 and not cfg.train_decoder:
        raise ValueError(
            "Encoder bootstrap requires decoder training. Enable --train_decoder or set --encoder_bootstrap_epochs 0."
        )

    # Device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        cfg.device = "cpu"

    device = torch.device(cfg.device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_config(cfg)
    run_logger = TrainingRunLogger(cfg)
    print("Config:", json.dumps(asdict(cfg), indent=2))
    if cfg.use_normals:
        print(
            "[train] --use_normals is currently a no-op for wi/wo. "
            "Sampled guide normals remain available only as training-side material data."
        )

    # Model
    model = NeuralMaterialModel(cfg).to(device)
    current_phase = get_training_phase(cfg, 0)
    opt = make_optimizer(model, cfg, current_phase)
    scheduler = make_scheduler(opt, cfg)

    best_val = float("inf")
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_epoch: Optional[int] = None
    best_phase: Optional[str] = None
    last_epoch: Optional[int] = None
    last_metrics: Optional[Dict[str, float]] = None
    run_status = "completed"

    try:
        # Gene validation data only once, and keep as single holdout set for all epochs
        data_generator = DataGenerator(sampleCount=cfg.validation_n)
        if cfg.encoder_bootstrap_epochs > 0 and not data_generator.supports_uv_grid():
            raise RuntimeError(
                "Encoder bootstrap requires the rebuilt OnlineDataGenerationPass plugin with UV-grid support. "
                "Rebuild Falcor/plugin binaries so setUvGrid/clearUvGrid are available, or set --encoder_bootstrap_epochs 0."
            )
        validation_batch = data_generator.generate_data(
            cfg.seed, SEED_DOMAIN_VALIDATION, 0
        ).copy()
        data_generator.release_data()
        validation_tensor = tensorize_batch(data_to_dict(validation_batch))
        print_first_sample(validation_tensor, "validation batch")

        data_generator = DataGenerator(sampleCount=cfg.training_n)
        if cfg.train_importance_sampler:
            print(f"[train] dual-decoder mode: shared batch={cfg.training_n}")
        else:
            print(f"[train] brdf-only mode: batch={cfg.training_n}")
        for epoch in range(cfg.max_epochs):
            phase = get_training_phase(cfg, epoch)
            phase_changed = phase != current_phase
            if phase_changed:
                print(f"[train] switching phase: {current_phase} -> {phase} at epoch {epoch:03d}")
                if phase == "finetune":
                    initialize_latent_texture_from_encoder(
                        model, cfg, epoch
                    )
                    for p in model.encoder.parameters():
                        p.requires_grad_(False)
                current_phase = phase

            maybe_freeze_parts(model, cfg, epoch=epoch)

            data_batch_decoder = data_generator.generate_data(
                cfg.seed, SEED_DOMAIN_TRAIN, epoch
            )
            training_tensor_decoder = tensorize_batch(data_to_dict(data_batch_decoder))

            training_tensor_sampler = None
            if cfg.train_importance_sampler:
                training_tensor_sampler = training_tensor_decoder

            if epoch == 0:
                print_first_sample(training_tensor_decoder, "training batch (decoder)")
                if training_tensor_sampler is not None:
                    print_first_sample(training_tensor_sampler, "training batch (sampler/shared)")

            train_metrics, opt, scheduler = train_one_epoch(
                model,
                training_tensor_decoder,
                training_tensor_sampler,
                opt,
                scheduler,
                cfg,
                epoch,
                phase,
            )

            scheduler.step()

            metrics = dict(train_metrics)

            val_metrics = validate(model, validation_tensor, cfg, epoch, phase)
            metrics.update(val_metrics)
            last_epoch = epoch
            last_metrics = dict(metrics)

            if cfg.print_every_epochs > 0 and (epoch % cfg.print_every_epochs == 0):
                elapsed = time.time() - run_start_time
                sampler_log = (
                    f" sampler_loss={metrics['sampler_loss']:.6f}"
                    if "sampler_loss" in metrics
                    else ""
                )
                print(
                    f"[train] epoch {epoch:03d} "
                    f"phase={phase} brdf_loss={metrics['brdf_loss']:.6f}" \
                    f"{sampler_log} "
                    f"val_loss={metrics['val_loss']:.6f} "
                    f"yhat_mean={metrics['yhat_mean']:.3e} "
                    f"elapsed={elapsed:.1f}s"
                )

            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                best_epoch = epoch
                best_phase = phase
                best_metrics = dict(metrics)
                best_model_state = snapshot_model_state(model)
                print(f"[best] epoch {epoch:03d} val_loss={best_val:.6f} cached in memory")

            if run_logger.should_log_progress(
                epoch=epoch,
                phase_changed=phase_changed,
                is_final=(epoch == cfg.max_epochs - 1),
            ):
                run_logger.append_progress(epoch, metrics, phase)

            data_generator.release_data()
    except KeyboardInterrupt:
        run_status = "interrupted"
        raise
    except Exception:
        run_status = "failed"
        raise
    finally:
        run_logger.write_summary(
            status=run_status,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            last_epoch=last_epoch,
            last_metrics=last_metrics,
        )

    if (
        best_model_state is not None
        and best_epoch is not None
        and best_metrics is not None
    ):
        model.load_state_dict(best_model_state)
        if best_phase == "bootstrap":
            initialize_latent_texture_from_encoder(
                model, cfg, best_epoch
            )
        best_ckpt_path = save_checkpoint(
            model,
            None,
            None,
            cfg,
            best_epoch,
            best_metrics,
            filename="best_checkpoint.pt",
        )
        print(
            f"[export] Restored best validation state from epoch "
            f"{best_epoch:03d} with val_loss={best_metrics.get('val_loss', float('nan')):.6f} "
            f"and saved {best_ckpt_path}"
        )

    export_renderer_assets(model, cfg)
    print("Done. Exports written to:", cfg.out_dir)


if __name__ == "__main__":
    main()
