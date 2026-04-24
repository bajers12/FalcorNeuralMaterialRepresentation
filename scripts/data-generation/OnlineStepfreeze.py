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
from DataGenerator import DataGenerator
from training_run_logging import TrainingRunLogger
import random

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
        return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

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
      - Decoder
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
    model: NeuralMaterialModel, cfg: TrainConfig, random_seed: int
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
            cfg.tex_w, cfg.tex_h, random_seed
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
    }
    min_by_name = {
        "latent": _latent_lr_min(cfg),
        "decoder": _decoder_lr_min(cfg),
        "encoder": _decoder_lr_min(cfg),
    }

    def lr_lambda_factory(group_name: str):
        base = base_by_name[group_name]
        min_lr = min_by_name[group_name]

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
    batch: Dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    epoch: int,
    phase: str,
    run_start_time: float,
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
    bsdf_loss = log_l1_loss(y_hat, y, cfg.log_eps)
    loss = bsdf_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()

    if cfg.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

    opt.step()

    with torch.no_grad():
        stats = compute_basic_stats(y_hat, y)
        raw_stats = compute_raw_stats(raw)

    return (
        {
            "loss": loss.item(),
            "bsdf_loss": bsdf_loss.item(),
            "phase": phase,
            "mae": stats["mae"],
            "yhat_mean": stats["yhat_mean"],
            "y_mean": stats["y_mean"],
            "raw_mean": raw_stats["raw_mean"],
            "raw_std": raw_stats["raw_std"],
        },
        opt,
        scheduler,
    )


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
    bsdf_loss = log_l1_loss(y_hat, y, cfg.log_eps)
    loss = bsdf_loss
    stats = compute_basic_stats(y_hat, y)
    raw_stats = compute_raw_stats(raw)

    out = {
        "phase": phase,
        "val_loss": loss.item(),
        "val_bsdf_loss": bsdf_loss.item(),
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
    filename: Optional[str] = None,
) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, f"checkpoint_epoch.pt")
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
        "exp_offset": np.array([cfg.exp_offset], dtype=np.float32),
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
    latent_ch = int(np.asarray(weights["latent_ch"]).reshape(-1)[0])
    num_frames = int(np.asarray(weights["num_frames"]).reshape(-1)[0])
    exp_offset = float(np.asarray(weights["exp_offset"]).reshape(-1)[0])

    ordered = [
        ("frame_linear.weight", (12, 8)),
        ("mlp.0.weight", (32, 20)),
        ("mlp.0.bias", (32,)),
        ("mlp.2.weight", (32, 32)),
        ("mlp.2.bias", (32,)),
        ("mlp.4.weight", (3, 32)),
        ("mlp.4.bias", (3,)),
    ]

    with open(path, "wb") as f:
        f.write(b"NMDLWT01")
        f.write(struct.pack("<iiif", latent_ch, num_frames, 1, exp_offset))

        for name, expected_shape in ordered:
            arr = np.asarray(weights[name], dtype=np.float32)
            if tuple(arr.shape) != expected_shape:
                raise ValueError(f"{name} expected shape {expected_shape}, got {arr.shape}")
            f.write(arr.astype(np.float32).ravel(order="C").tobytes())


def save_metadata(path: Path, latent: np.ndarray, weights: dict) -> None:
    _, h, w = latent.shape
    metadata = {
        "width": int(w),
        "height": int(h),
        "latent_dim": int(latent.shape[0]),
        "num_frames": int(np.asarray(weights["num_frames"]).reshape(-1)[0]),
        "exp_offset": float(np.asarray(weights["exp_offset"]).reshape(-1)[0]),
        "apply_exp": True,
        "decoder_layout": {
            "frame_linear.weight": [12, 8],
            "mlp.0.weight": [32, 20],
            "mlp.0.bias": [32],
            "mlp.2.weight": [32, 32],
            "mlp.2.bias": [32],
            "mlp.4.weight": [3, 32],
            "mlp.4.bias": [3],
        },
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
        "exp_offset": np.array([cfg.exp_offset], dtype=np.float32),
        "frame_linear.weight": sd["frame_linear.weight"].detach().cpu().numpy(),
        "mlp.0.weight": sd["mlp.0.weight"].detach().cpu().numpy(),
        "mlp.0.bias": sd["mlp.0.bias"].detach().cpu().numpy(),
        "mlp.2.weight": sd["mlp.2.weight"].detach().cpu().numpy(),
        "mlp.2.bias": sd["mlp.2.bias"].detach().cpu().numpy(),
        "mlp.4.weight": sd["mlp.4.weight"].detach().cpu().numpy(),
        "mlp.4.bias": sd["mlp.4.bias"].detach().cpu().numpy(),
    }

    save_weights_bin(preview_dir / "decoder_weights.bin", weights)
    save_metadata(preview_dir / "metadata.json", latent, weights)

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
    p.add_argument("--mlp_width", type=int, default=64)
    p.add_argument("--mlp_depth", type=int, default=2)
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
    p.add_argument("--train_latent_texture", action="store_true")
    p.add_argument("--no_train_latent_texture", action="store_true")
    p.add_argument("--train_decoder", action="store_true")
    p.add_argument("--no_train_decoder", action="store_true")

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
        "--no_albedo_feature",
        action="store_true",
        help="Exclude albedo from the training-only material encoder.",
    )
    p.add_argument(
        "--no_spec_feature",
        action="store_true",
        help="Exclude specular from the training-only material encoder.",
    )
    p.add_argument(
        "--no_normal_feature",
        action="store_true",
        help="Exclude guide normal from the training-only material encoder.",
    )
    p.add_argument(
        "--no_roughness_feature",
        action="store_true",
        help="Exclude roughness from the training-only material encoder.",
    )
    p.add_argument(
        "--no_pdf_feature",
        action="store_true",
        help="Exclude BSDF pdf from the training-only material encoder.",
    )
    p.add_argument(
        "--use_pdf_feature",
        action="store_true",
        help="Explicitly include BSDF pdf in the training-only material encoder.",
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
    if args.no_train_latent_texture:
        cfg.train_latent_texture = False
    elif args.train_latent_texture:
        cfg.train_latent_texture = True

    if args.no_train_decoder:
        cfg.train_decoder = False
    elif args.train_decoder:
        cfg.train_decoder = True

    cfg.freeze_latent_after_epoch = args.freeze_latent_after_epoch
    cfg.freeze_decoder_after_epoch = args.freeze_decoder_after_epoch

    cfg.use_normals = args.use_normals
    cfg.encoder_width = args.encoder_width
    cfg.encoder_depth = args.encoder_depth
    cfg.encoder_bootstrap_epochs = max(0, args.encoder_bootstrap_epochs)
    cfg.latent_init_batch_size = max(1, args.latent_init_batch_size)
    cfg.use_albedo_features = not args.no_albedo_feature
    cfg.use_spec_features = not args.no_spec_feature
    cfg.use_normal_features = not args.no_normal_feature
    cfg.use_roughness_feature = not args.no_roughness_feature
    cfg.use_pdf_feature = cfg.use_pdf_feature
    if args.use_pdf_feature:
        cfg.use_pdf_feature = True
    if args.no_pdf_feature:
        cfg.use_pdf_feature = False

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
        validation_batch = data_generator.generate_data(random.randint(0, 1000000)).copy()
        data_generator.release_data()
        validation_tensor = tensorize_batch(data_to_dict(validation_batch))
        print_first_sample(validation_tensor, "validation batch")

        data_generator = DataGenerator(sampleCount=cfg.training_n)
        for epoch in range(cfg.max_epochs):
            phase = get_training_phase(cfg, epoch)
            phase_changed = phase != current_phase
            if phase_changed:
                print(f"[train] switching phase: {current_phase} -> {phase} at epoch {epoch:03d}")
                if phase == "finetune":
                    initialize_latent_texture_from_encoder(
                        model, cfg, random.randint(0, 1000000)
                    )
                    for p in model.encoder.parameters():
                        p.requires_grad_(False)
                current_phase = phase

            maybe_freeze_parts(model, cfg, epoch=epoch)

            data_batch = data_generator.generate_data(random.randint(0, 1000000))
            training_batch = data_batch
            training_tensor = tensorize_batch(data_to_dict(training_batch))
            if epoch == 0:
                print_first_sample(training_tensor, "training batch")

            train_metrics, opt, scheduler = train_one_epoch(
                model,
                training_tensor,
                opt,
                scheduler,
                cfg,
                epoch,
                phase,
                run_start_time,
            )

            scheduler.step()

            metrics = dict(train_metrics)

            val_metrics = validate(model, validation_tensor, cfg, epoch, phase)
            metrics.update(val_metrics)
            last_epoch = epoch
            last_metrics = dict(metrics)

            if cfg.print_every_epochs > 0 and (epoch % cfg.print_every_epochs == 0):
                elapsed = time.time() - run_start_time
                print(
                    f"[train] epoch {epoch:03d} "
                    f"phase={phase} train_loss={metrics['loss']:.6f} "
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
                model, cfg, random.randint(0, 1000000)
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
