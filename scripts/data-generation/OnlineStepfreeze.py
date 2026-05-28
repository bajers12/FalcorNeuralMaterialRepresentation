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
from dataclasses import dataclass, asdict, field
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
import AssetConverter
from BrdfDecoder import Decoder
from ImportanceSamplingDecoder import ImportanceSamplingDecoder
from MaterialEncoder import MaterialEncoder
from LatentTexture import LatentTexture


# =============================================================================
# Config
# =============================================================================


@dataclass
class TrainConfig:
    # Latent texture
    tex_h: int = 512
    tex_w: int = 512
    latent_ch: int = 8
    hierarchical_mip_count: int = 5
    mip_exponential_rate: float = 0.7
    min_filter_sample_count: int = 1
    max_filter_sample_count: int = 8
    gaussian_filter_std_scale: float = 0.5

    scene_path: str = 'MatXScenes/Preview/MatXScene.pyscene'

    # Decoder architecture
    num_frames: int = 2
    brdf_mlp_width: int = 32
    brdf_mlp_depth: int = 2  # number of hidden layers
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
    max_epochs: int = 300000
    sampler_epochs: int = 20000

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

    # Paper-style target mollification for early BRDF decoder training.
    enable_mollification: bool = False
    mollification_start_angle_deg: float = 10.0
    mollification_iterations: int = 20000
    mollification_sample_count: int = 256

    # Legacy flag retained for CLI compatibility. Direction transforms are disabled;
    # sampled normals remain available only as training-side material features.
    use_normals: bool = False

    # Training-only encoder that maps sampled material values to latent codes.
    encoder_width: int = 64
    encoder_depth: int = 4
    encoder_bootstrap_epochs: int = 200
    latent_init_batch_size: int = 65536
    bootstrap_feature_layout: str = "auto"
    material_feature_dim: int = 0
    material_feature_names: Tuple[str, ...] = field(default_factory=tuple)
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
    ordered_keys = ["uv", "wi", "wo", "y", "mip_level", "features"]
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
    if cfg.material_feature_dim > 0:
        return cfg.material_feature_dim
    if cfg.encoder_bootstrap_epochs <= 0:
        return 1
    raise ValueError("Encoder bootstrap requires an active feature layout. Use --bootstrap_feature_layout auto, material, or legacy.")


# =============================================================================
# Model
# =============================================================================

class NeuralMaterialModel(nn.Module):
    """
    Wraps:
        - LatentTexture
        - Decoder (BRDF evaluation)
        - ImportanceSamplingDecoder (direction sampling)
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.latent = LatentTexture(cfg.tex_h, cfg.tex_w, cfg.latent_ch, cfg.hierarchical_mip_count)
        self.decoder = Decoder(
            latent_ch=cfg.latent_ch,
            num_frames=cfg.num_frames,
            mlp_width=cfg.brdf_mlp_width,
            mlp_depth=cfg.brdf_mlp_depth,
            use_bias_in_mlp=cfg.use_bias_in_mlp,
            frame_linear_bias=cfg.frame_linear_bias,
            exp_offset=cfg.exp_offset,
        )
        self.importance_sampler = ImportanceSamplingDecoder(
            latent_ch=cfg.latent_ch,
            mlp_width=cfg.sampler_mlp_width,
            mlp_depth=cfg.sampler_mlp_depth,
            use_bias_in_mlp=cfg.use_bias_in_mlp,
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
        self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, mip_level: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.latent.sample(uv, mip_level)
        return self.decode_with_raw(z, wi, wo)

    def forward(
        self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, mip_level: torch.Tensor | None = None
    ) -> torch.Tensor:
        y_hat, _raw = self.forward_with_raw(uv, wi, wo, mip_level)
        return y_hat


# =============================================================================
# Loss / Metrics
# =============================================================================


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


def compute_mip_validation_losses(
    raw: torch.Tensor,
    y: torch.Tensor,
    mip: Optional[torch.Tensor],
    cfg: TrainConfig,
    exp_offset: float,
) -> Dict[str, float]:
    if mip is None or cfg.hierarchical_mip_count <= 1:
        return {}

    mip_index = mip.reshape(-1).long().clamp(0, cfg.hierarchical_mip_count - 1)
    out: Dict[str, float] = {}
    for mip_level in range(cfg.hierarchical_mip_count):
        mask = mip_index == mip_level
        sample_count = int(mask.sum().item())
        out[f"brdf_val_count_mip{mip_level}"] = float(sample_count)
        if sample_count == 0:
            continue
        mip_loss = Decoder.log_l1_loss(
            raw[mask],
            y[mask],
            exp_offset,
            cfg.log_eps,
        )
        out[f"brdf_val_loss_mip{mip_level}"] = mip_loss.item()
    return out


def build_material_features(
    batch: Dict[str, torch.Tensor], cfg: TrainConfig, device: torch.device
) -> torch.Tensor:
    if "features" not in batch:
        raise ValueError("Configured bootstrap features are missing from the generated batch.")
    features = batch["features"].to(device, non_blocking=True)
    mean = getattr(cfg, "_bootstrap_feature_mean", None)
    std = getattr(cfg, "_bootstrap_feature_std", None)
    if mean is not None and std is not None:
        features = (features - mean) / std
    return features


def fit_bootstrap_feature_normalization(
    batch: Dict[str, torch.Tensor], cfg: TrainConfig, device: torch.device
) -> None:
    if "features" not in batch:
        return
    features = batch["features"].to(device, non_blocking=True)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    cfg._bootstrap_feature_mean = mean
    cfg._bootstrap_feature_std = std
    print(
        "[bootstrap] normalized encoder features: "
        f"mean_range=[{mean.min().item():.3e}, {mean.max().item():.3e}], "
        f"std_range=[{std.min().item():.3e}, {std.max().item():.3e}]"
    )


def get_training_phase(cfg: TrainConfig, epoch: int) -> str:
    if epoch < cfg.encoder_bootstrap_epochs:
        return "bootstrap"
    return "finetune"


def get_mollification_cone_angle_rad(cfg: TrainConfig, iteration: int) -> float:
    if (
        not cfg.enable_mollification
        or cfg.mollification_sample_count <= 1
        or cfg.mollification_start_angle_deg <= 0.0
        or cfg.mollification_iterations <= 0
        or iteration >= cfg.mollification_iterations
    ):
        return 0.0

    t = max(0.0, min(float(iteration) / float(cfg.mollification_iterations), 1.0))
    angle_deg = 0.5 * cfg.mollification_start_angle_deg * (1.0 + math.cos(math.pi * t))
    return math.radians(angle_deg)


def latent_init_filter_sample_count(cfg: TrainConfig, mip_level: int) -> int:
    if cfg.hierarchical_mip_count <= 1 or mip_level == 0:
        return 1
    count = 1 << min(30, 2 * mip_level)
    return int(
        max(
            cfg.min_filter_sample_count,
            min(count, max(cfg.min_filter_sample_count, cfg.max_filter_sample_count)),
        )
    )


def make_latent_init_uv_samples(
    level_w: int,
    level_h: int,
    mip_level: int,
    start_texel: int,
    texel_count: int,
    filter_sample_count: int,
    cfg: TrainConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    texel_indices = np.arange(start_texel, start_texel + texel_count, dtype=np.int64)
    texel_x = texel_indices % level_w
    texel_y = texel_indices // level_w
    centers = np.stack(
        [
            (texel_x.astype(np.float32) + 0.5) / float(level_w),
            (texel_y.astype(np.float32) + 0.5) / float(level_h),
        ],
        axis=1,
    )

    if mip_level == 0 or filter_sample_count <= 1 or cfg.gaussian_filter_std_scale <= 0.0:
        return centers.astype(np.float32, copy=False)

    centers = np.repeat(centers[:, None, :], filter_sample_count, axis=1)
    footprint = float(2**mip_level)
    sigma = np.array(
        [
            cfg.gaussian_filter_std_scale * footprint / max(float(cfg.tex_w), 1.0),
            cfg.gaussian_filter_std_scale * footprint / max(float(cfg.tex_h), 1.0),
        ],
        dtype=np.float32,
    )
    offsets = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(texel_count, filter_sample_count, 2),
    ).astype(np.float32)
    return np.mod(centers + offsets * sigma, 1.0).reshape(-1, 2).astype(np.float32, copy=False)


@torch.no_grad()
def initialize_latent_texture_from_encoder(
    model: NeuralMaterialModel, cfg: TrainConfig, generation_index: int
) -> None:
    if get_encoder_input_dim(cfg) == 0:
        raise ValueError("Cannot initialize latent texture without encoder features.")

    device = torch.device(cfg.device)
    model.eval()

    print(
        f"[bootstrap] Initializing {cfg.hierarchical_mip_count} latent mip(s) "
        f"from encoder, finest grid={cfg.tex_w}x{cfg.tex_h}"
    )

    for mip_level, latent_level in enumerate(model.latent.levels):
        level_h, level_w = model.latent.level_shape(mip_level)
        sample_count = level_w * level_h
        filter_sample_count = latent_init_filter_sample_count(cfg, mip_level)
        texels_per_chunk = max(1, cfg.latent_init_batch_size // filter_sample_count)
        uv_sample_capacity = texels_per_chunk * filter_sample_count
        rng = np.random.default_rng(
            (
                int(cfg.seed)
                ^ (0x9E3779B9 * (generation_index + 1))
                ^ (0x85EBCA6B * (mip_level + 1))
            )
            & 0xFFFFFFFF
        )
        print(
            f"[bootstrap] mip {mip_level}: grid={level_w}x{level_h}, "
            f"feature_samples_per_texel={filter_sample_count}"
        )
        grid_generator = DataGenerator(
            sampleCount=uv_sample_capacity,
            bootstrap_feature_layout=cfg.bootstrap_feature_layout,
            scene_path=cfg.scene_path,
            hierarchical_filtering_enabled=False,
        )
        try:
            if not grid_generator.supports_uv_samples():
                raise RuntimeError(
                    "Coarse latent initialization requires OnlineDataGenerationPass.setUvSamples. "
                    "Rebuild Falcor/plugin binaries so arbitrary bootstrap UV sampling is available."
                )

            latent_image_flat = torch.empty((sample_count, cfg.latent_ch), dtype=torch.float32)
            for start_texel in range(0, sample_count, texels_per_chunk):
                texel_count = min(texels_per_chunk, sample_count - start_texel)
                uv_samples = make_latent_init_uv_samples(
                    level_w,
                    level_h,
                    mip_level,
                    start_texel,
                    texel_count,
                    filter_sample_count,
                    cfg,
                    rng,
                )
                actual_uv_count = uv_samples.shape[0]
                if actual_uv_count < uv_sample_capacity:
                    pad_count = uv_sample_capacity - actual_uv_count
                    uv_samples = np.concatenate(
                        [uv_samples, np.repeat(uv_samples[-1:], pad_count, axis=0)],
                        axis=0,
                    )

                grid_batch = grid_generator.generate_uv_data(
                    uv_samples,
                    cfg.seed,
                    SEED_DOMAIN_BOOTSTRAP,
                    generation_index + mip_level,
                ).copy()
                grid_generator.release_data()

                grid_tensor = tensorize_batch(data_to_dict(grid_batch[:actual_uv_count], cfg.material_feature_dim))
                latent_chunks = []

                for start in range(0, actual_uv_count, cfg.latent_init_batch_size):
                    end = min(start + cfg.latent_init_batch_size, actual_uv_count)
                    chunk = {key: value[start:end] for key, value in grid_tensor.items()}
                    features = build_material_features(chunk, cfg, device)
                    latent_chunks.append(model.encoder(features).cpu())

                latent_values = torch.cat(latent_chunks, dim=0)
                if filter_sample_count > 1:
                    latent_values = latent_values.view(texel_count, filter_sample_count, cfg.latent_ch).mean(dim=1)
                latent_image_flat[start_texel : start_texel + texel_count] = latent_values
        finally:
            grid_generator.release_data()

        latent_image = latent_image_flat.view(
            level_h, level_w, cfg.latent_ch
        )

        z_image = latent_image.permute(2, 0, 1).unsqueeze(0).contiguous()
        latent_level.copy_(z_image.to(device))


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
    """BRDF/latent/encoder optimizer — never includes sampler params."""
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


def make_sampler_optimizer(
    model: NeuralMaterialModel, cfg: TrainConfig
) -> torch.optim.Optimizer:
    """
    Dedicated optimizer for the importance sampler — completely isolated from BRDF optimizer.
    """
    sampler_params = [p for p in model.importance_sampler.parameters() if p.requires_grad]
    if not sampler_params:
        raise ValueError("Importance sampler has no trainable parameters.")
    return torch.optim.Adam(
        [{"params": sampler_params, "lr": _decoder_lr(cfg), "name": "sampler"}],
        weight_decay=cfg.weight_decay,
    )


def make_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    """
    Cosine LR decay from the per-group base LR to the per-group minimum over cfg.max_epochs (epoch-stepped).
    Used for the BRDF/latent/encoder optimizer only.
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


def make_sampler_scheduler(sampler_opt: torch.optim.Optimizer, cfg: TrainConfig):
    """
    Cosine LR decay for the dedicated sampler optimizer, decaying over sampler_epochs
    (not max_epochs) so the sampler gets a full cosine cycle within its training window.
    """
    base = _decoder_lr(cfg)
    min_lr = _decoder_lr_min(cfg)
    total = max(cfg.sampler_epochs - 1, 1)

    def lr_lambda(epoch: int):
        t = min(epoch / total, 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * t))
        lr_now = min_lr + (base - min_lr) * scale
        return lr_now / max(base, 1e-12)

    return torch.optim.lr_scheduler.LambdaLR(sampler_opt, lr_lambda)


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
    """Rebuild BRDF/latent/encoder optimizer when requires_grad state changes.
    The sampler optimizer is managed separately and never included here.
    """
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
    # NOTE: sampler is intentionally excluded — it has its own optimizer.

    current_group_names = [pg.get("name") for pg in opt.param_groups]
    if active_group_names == current_group_names:
        return opt, scheduler

    # Transfer Adam moments to the rebuilt optimizer
    old_state = opt.state

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
    batch_decoder: Dict[str, torch.Tensor],
    batch_sampler: Optional[Dict[str, torch.Tensor]],
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    epoch: int,
    phase: str,
    sampler_opt: Optional[torch.optim.Optimizer] = None,
    sampler_scheduler=None,
):
    model.train()
    device = torch.device(cfg.device)
    latent_frozen_logged = False

    opt, scheduler = maybe_rebuild_optimizer_and_scheduler(
        model, opt, scheduler, cfg, phase
    )

    latent_now_frozen = all(not p.requires_grad for p in model.latent.parameters())
    if latent_now_frozen and not latent_frozen_logged:
        print(f"[train] freezing latent texture at epoch={epoch}")
        latent_frozen_logged = True

    # ------------------------------------------------------------------ #
    #  BRDF decoder step                                                   #
    # ------------------------------------------------------------------ #
    uv_dec = batch_decoder["uv"].to(device, non_blocking=True)
    y_dec = batch_decoder["y"].to(device, non_blocking=True)

    wi_dec, wo_dec = _maybe_transform_dirs_with_normals(batch_decoder, cfg, device)
    if cfg.clamp_min_target > 0.0:
        y_dec = y_dec.clamp_min(cfg.clamp_min_target)
    opt.zero_grad(set_to_none=True)

    if phase == "bootstrap":
        material_features_dec = build_material_features(batch_decoder, cfg, device)
        z_dec = model.encoder(material_features_dec)
    else:
        mip_dec = batch_decoder.get("mip_level")
        mip_dec = mip_dec.to(device, non_blocking=True) if mip_dec is not None else None
        z_dec = model.latent.sample(uv_dec, mip_dec)

    y_hat_dec, raw_dec = model.decode_with_raw(z_dec, wi_dec, wo_dec)
    bsdf_loss = Decoder.log_l1_loss(raw_dec, y_dec, model.decoder.exp_offset, cfg.log_eps)

    if not torch.isfinite(bsdf_loss):
        raise RuntimeError(f"Non-finite BRDF loss at epoch {epoch}: {bsdf_loss.item()}")

    bsdf_loss.backward()

    if cfg.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(
            [p for pg in opt.param_groups for p in pg["params"] if p.grad is not None],
            cfg.grad_clip_norm,
        )

    opt.step()
    scheduler.step()

    # ------------------------------------------------------------------ #
    #  Importance sampler step (separate optimizer)                        #
    # ------------------------------------------------------------------ #
    sampler_loss = None

    should_train_sampler = (
        cfg.train_importance_sampler
        and sampler_opt is not None
        and (epoch < cfg.sampler_epochs + cfg.encoder_bootstrap_epochs)
        and (epoch >= cfg.encoder_bootstrap_epochs)
    )
    if should_train_sampler:
        sampler_batch = batch_decoder if batch_sampler is None else batch_sampler

        uv_sam = sampler_batch["uv"].to(device, non_blocking=True)
        wi_sam, _ = _maybe_transform_dirs_with_normals(sampler_batch, cfg, device)
        sampler_opt.zero_grad(set_to_none=True)

        mip_sam = sampler_batch.get("mip_level")
        mip_sam = mip_sam.to(device, non_blocking=True) if mip_sam is not None else None
        z_sam = model.latent.sample(uv_sam, mip_sam).detach()

        # KL-divergence loss (paper Section 4.3).
        # z_sam is detached: latent has no grad w.r.t. sampler (paper stability trick).
        pred = model.importance_sampler(z_sam, wi_sam);
        sampler_loss = model.importance_sampler.loss(pred, model.decoder, z_sam, wi_sam, cfg.log_eps)
        if not torch.isfinite(sampler_loss):
            raise RuntimeError(f"Non-finite sampler loss at epoch {epoch}: {sampler_loss.item()}")

        sampler_loss.backward()

        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                [p for p in model.importance_sampler.parameters() if p.grad is not None],
                cfg.grad_clip_norm,
            )

        sampler_opt.step()
        sampler_scheduler.step()

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

    mip = batch.get("mip_level")
    mip = mip.to(device, non_blocking=True) if mip is not None else None

    if phase == "bootstrap":
        material_features = build_material_features(batch, cfg, device)
        z = model.encoder(material_features)
    else:
        z = model.latent.sample(uv, mip)

    y_hat, raw = model.decode_with_raw(z, wi, wo)
    loss = Decoder.log_l1_loss(raw, y, model.decoder.exp_offset, cfg.log_eps)
    stats = compute_basic_stats(y_hat, y)
    raw_stats = compute_raw_stats(raw)
    mip_metrics = (
        compute_mip_validation_losses(raw, y, mip, cfg, model.decoder.exp_offset)
        if phase != "bootstrap"
        else {}
    )

    out = {
        "phase": phase,
        "val_loss": loss.item(),
        "brdf_val_loss": loss.item(),
        "val_mae": stats["mae"],
        "val_yhat_mean": stats["yhat_mean"],
        "val_y_mean": stats["y_mean"],
        "val_raw_mean": raw_stats["raw_mean"],
        "val_raw_std": raw_stats["raw_std"],
    }
    out.update(mip_metrics)
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
    }
    torch.save(payload, ckpt_path)
    return ckpt_path

def save_config(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()

    p.add_argument("--scene_path", type=str, default='MatXScenes/Preview/MatXScene.pyscene')
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
    p.add_argument(
        "--hierarchical_mip_count",
        type=int,
        default=5,
        help="Number of independently trained latent mip levels, starting from tex_w/tex_h.",
    )
    p.add_argument(
        "--mip_exponential_rate",
        type=float,
        default=0.7,
        help="Truncated exponential rate used to randomize training mip levels.",
    )
    p.add_argument("--min_filter_sample_count", type=int, default=1)
    p.add_argument("--max_filter_sample_count", type=int, default=8)
    p.add_argument(
        "--gaussian_filter_std_scale",
        type=float,
        default=0.5,
        help="Spatial Gaussian footprint sigma in texels, scaled by the selected mip footprint.",
    )

    p.add_argument("--num_frames", type=int, default=2)
    p.add_argument("--brdf_mlp_width", type=int, default=64)
    p.add_argument("--brdf_mlp_depth", type=int, default=2)
    p.add_argument("--sampler_mlp_width", type=int, default=32)
    p.add_argument("--sampler_mlp_depth", type=int, default=3)
    p.add_argument("--exp_offset", type=float, default=3.0)

    p.add_argument("--training_n", type=int, default=65536)
    p.add_argument("--validation_size", type=int, default=65536)
    p.add_argument("--max_epochs", type=int, default=300000)
    p.add_argument("--sampler_epochs", type=int, default=20000)
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
        "--enable_mollification",
        action="store_true",
        help="Blur early BRDF targets by averaging outgoing directions in a shrinking cone around wo.",
    )
    p.add_argument("--mollification_start_angle_deg", type=float, default=10.0)
    p.add_argument("--mollification_iterations", type=int, default=20000)
    p.add_argument("--mollification_sample_count", type=int, default=256)

    p.add_argument(
        "--use_normals",
        action="store_true",
        help="Legacy no-op kept for CLI compatibility. Sampled guide normals stay on the training/material side only.",
    )
    p.add_argument("--encoder_width", type=int, default=64)
    p.add_argument("--encoder_depth", type=int, default=4)
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
        "--bootstrap_feature_layout",
        type=str,
        default="auto",
        choices=("none", "auto", "legacy", "material", "three_layered_ggx"),
        help="Bootstrap-only encoder feature layout. Finetune generation always uses 'none'. 'three_layered_ggx' is kept as an alias for 'material'.",
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
    cfg.scene_path = args.scene_path

    cfg.tex_h = args.tex_h
    cfg.tex_w = args.tex_w
    cfg.latent_ch = args.latent_ch
    cfg.hierarchical_mip_count = max(1, args.hierarchical_mip_count)
    cfg.mip_exponential_rate = max(1e-6, args.mip_exponential_rate)
    cfg.min_filter_sample_count = max(1, args.min_filter_sample_count)
    cfg.max_filter_sample_count = max(cfg.min_filter_sample_count, args.max_filter_sample_count)
    cfg.gaussian_filter_std_scale = max(0.0, args.gaussian_filter_std_scale)

    cfg.num_frames = args.num_frames
    cfg.brdf_mlp_width = args.brdf_mlp_width
    cfg.brdf_mlp_depth = args.brdf_mlp_depth
    cfg.sampler_mlp_width = args.sampler_mlp_width
    cfg.sampler_mlp_depth = args.sampler_mlp_depth
    cfg.exp_offset = args.exp_offset

    cfg.training_n = args.training_n
    cfg.validation_n = args.validation_size
    cfg.max_epochs = args.max_epochs
    cfg.sampler_epochs = args.sampler_epochs
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
    cfg.enable_mollification = args.enable_mollification
    cfg.mollification_start_angle_deg = max(0.0, args.mollification_start_angle_deg)
    cfg.mollification_iterations = max(0, args.mollification_iterations)
    cfg.mollification_sample_count = max(1, args.mollification_sample_count)

    cfg.use_normals = args.use_normals
    cfg.encoder_width = args.encoder_width
    cfg.encoder_depth = args.encoder_depth
    cfg.encoder_bootstrap_epochs = max(0, args.encoder_bootstrap_epochs)
    cfg.latent_init_batch_size = max(1, args.latent_init_batch_size)
    cfg.bootstrap_feature_layout = args.bootstrap_feature_layout
    cfg.use_albedo_features = args.albedo_feature
    cfg.use_spec_features = args.spec_feature
    cfg.use_normal_features = args.normal_feature
    cfg.use_roughness_feature = args.roughness_feature
    cfg.use_pdf_feature = args.pdf_feature

    cfg.train_importance_sampler = args.train_importance_sampler

    return cfg


def data_to_dict(data: np.ndarray, material_feature_dim: int = 0):
    uv = data[:, 0:2]
    wo = data[:, 2:5]
    wi = data[:, 5:8]
    f = data[:, 8:11]
    has_mip_column = data.shape[1] >= (12 + material_feature_dim)
    mip_level = data[:, 11:12] if has_mip_column else np.zeros((data.shape[0], 1), dtype=data.dtype)

    batch = {
        "uv": uv,
        "wo": wo,
        "wi": wi,
        "y": f,
        "mip_level": mip_level.reshape(-1),
    }
    if material_feature_dim > 0:
        feature_start = 12 if has_mip_column else 11
        feature_end = feature_start + material_feature_dim
        if data.shape[1] < feature_end:
            raise ValueError(
                f"Generated data has {data.shape[1]} columns, but material feature dim "
                f"{material_feature_dim} requires at least {feature_end}."
            )
        batch["features"] = data[:, feature_start:feature_end]

    return batch

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

    bootstrap_validation_generator = None
    if cfg.encoder_bootstrap_epochs > 0:
        bootstrap_validation_generator = DataGenerator(
            sampleCount=cfg.validation_n,
            bootstrap_feature_layout=cfg.bootstrap_feature_layout,
            scene_path=cfg.scene_path,
            hierarchical_filtering_enabled=False,
        )
        cfg.material_feature_names = tuple(bootstrap_validation_generator.get_bootstrap_feature_names())
        cfg.material_feature_dim = bootstrap_validation_generator.get_bootstrap_feature_dim()
        if cfg.material_feature_dim != len(cfg.material_feature_names):
            raise RuntimeError(
                "OnlineDataGenerationPass reported inconsistent bootstrap feature metadata: "
                f"dim={cfg.material_feature_dim}, names={len(cfg.material_feature_names)}."
            )
        if cfg.material_feature_dim <= 0:
            raise RuntimeError(
                "Encoder bootstrap is enabled, but the selected bootstrap feature layout produced no features."
            )
        print(
            "[bootstrap] using encoder features: "
            f"layout={cfg.bootstrap_feature_layout}, dim={cfg.material_feature_dim}"
        )

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

    sampler_opt = None
    sampler_scheduler = None

    best_brdf_val_loss = float("inf")
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_epoch: Optional[int] = None
    best_phase: Optional[str] = None
    best_bootstrap_val_loss = float("inf")
    best_bootstrap_state: Optional[Dict[str, torch.Tensor]] = None
    best_bootstrap_metrics: Optional[Dict[str, float]] = None
    best_bootstrap_epoch: Optional[int] = None
    best_finetune_val_loss = float("inf")
    best_finetune_state: Optional[Dict[str, torch.Tensor]] = None
    best_finetune_metrics: Optional[Dict[str, float]] = None
    best_finetune_epoch: Optional[int] = None
    last_epoch: Optional[int] = None
    last_metrics: Optional[Dict[str, float]] = None
    run_status = "completed"

    best_sampler_loss = float("inf")
    best_sampler_state: Optional[Dict[str, torch.Tensor]] = None
    best_sampler_epoch: Optional[int] = None

    try:
        # Generate validation data once per required payload shape.
        bootstrap_validation_tensor = None
        if bootstrap_validation_generator is not None:
            if not bootstrap_validation_generator.supports_uv_grid():
                raise RuntimeError(
                    "Encoder bootstrap requires the rebuilt OnlineDataGenerationPass plugin with UV-grid support. "
                    "Rebuild Falcor/plugin binaries so setUvGrid/clearUvGrid are available, or set --encoder_bootstrap_epochs 0."
                )
            bootstrap_validation_batch = bootstrap_validation_generator.generate_data(
                cfg.seed, SEED_DOMAIN_VALIDATION, 0
            ).copy()
            bootstrap_validation_generator.release_data()
            bootstrap_validation_tensor = tensorize_batch(data_to_dict(bootstrap_validation_batch, cfg.material_feature_dim))
            fit_bootstrap_feature_normalization(bootstrap_validation_tensor, cfg, device)
            print_first_sample(bootstrap_validation_tensor, "bootstrap validation batch")

        validation_generator = DataGenerator(
            sampleCount=cfg.validation_n,
            bootstrap_feature_layout="none",
            scene_path=cfg.scene_path,
            hierarchical_filtering_enabled=cfg.hierarchical_mip_count > 1,
            hierarchical_mip_count=cfg.hierarchical_mip_count,
            finest_texture_width=cfg.tex_w,
            finest_texture_height=cfg.tex_h,
            mip_exponential_rate=cfg.mip_exponential_rate,
            min_filter_sample_count=cfg.min_filter_sample_count,
            max_filter_sample_count=cfg.max_filter_sample_count,
            gaussian_filter_std_scale=cfg.gaussian_filter_std_scale,
        )
        if cfg.encoder_bootstrap_epochs > 0 and not validation_generator.supports_uv_grid():
            raise RuntimeError(
                "Encoder bootstrap requires the rebuilt OnlineDataGenerationPass plugin with UV-grid support. "
                "Rebuild Falcor/plugin binaries so setUvGrid/clearUvGrid are available, or set --encoder_bootstrap_epochs 0."
            )
        validation_batch = validation_generator.generate_data(
            cfg.seed, SEED_DOMAIN_VALIDATION, 0
        ).copy()
        validation_generator.release_data()
        validation_tensor = tensorize_batch(data_to_dict(validation_batch, 0))
        print_first_sample(validation_tensor, "finetune validation batch")

        bootstrap_data_generator = None
        if cfg.encoder_bootstrap_epochs > 0:
            bootstrap_data_generator = DataGenerator(
                sampleCount=cfg.training_n,
                bootstrap_feature_layout=cfg.bootstrap_feature_layout,
                scene_path=cfg.scene_path,
                hierarchical_filtering_enabled=False,
            )
        finetune_data_generator = DataGenerator(
            sampleCount=cfg.training_n,
            bootstrap_feature_layout="none",
            scene_path=cfg.scene_path,
            hierarchical_filtering_enabled=cfg.hierarchical_mip_count > 1,
            hierarchical_mip_count=cfg.hierarchical_mip_count,
            finest_texture_width=cfg.tex_w,
            finest_texture_height=cfg.tex_h,
            mip_exponential_rate=cfg.mip_exponential_rate,
            min_filter_sample_count=cfg.min_filter_sample_count,
            max_filter_sample_count=cfg.max_filter_sample_count,
            gaussian_filter_std_scale=cfg.gaussian_filter_std_scale,
        )
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
                    if best_bootstrap_state is not None:
                        model.load_state_dict(best_bootstrap_state)
                        print(
                            f"[bootstrap] restored best bootstrap state from epoch "
                            f"{best_bootstrap_epoch:03d} before latent initialization "
                            f"(val_loss={best_bootstrap_val_loss:.6f})"
                        )
                    initialize_latent_texture_from_encoder(
                        model, cfg, epoch
                    )
                    for p in model.encoder.parameters():
                        p.requires_grad_(False)

                    post_mean = model.latent.Z.mean().item()
                    post_std = model.latent.Z.std().item()
                    print(f"[train] latent post-init mean={post_mean:.6e} std={post_std:.6e}")


                    # Recreate sampler optimizer/scheduler so Adam moments don't slow adaptation
                    if cfg.train_importance_sampler:
                        sampler_opt = make_sampler_optimizer(model, cfg)
                        sampler_scheduler = make_sampler_scheduler(sampler_opt, cfg)
                        print("[train] reinitialized sampler optimizer and scheduler after latent bootstrap")
                current_phase = phase

            maybe_freeze_parts(model, cfg, epoch=epoch)

            mollification_cone_angle_rad = get_mollification_cone_angle_rad(cfg, epoch)
            if epoch == 0 and cfg.enable_mollification:
                print(
                    "[train] mollification enabled: "
                    f"start_angle={cfg.mollification_start_angle_deg:.3f} deg, "
                    f"iterations={cfg.mollification_iterations}, "
                    f"samples={cfg.mollification_sample_count}"
                )
            active_data_generator = bootstrap_data_generator if phase == "bootstrap" else finetune_data_generator
            active_feature_dim = cfg.material_feature_dim if phase == "bootstrap" else 0
            data_batch_decoder = active_data_generator.generate_data(
                cfg.seed,
                SEED_DOMAIN_TRAIN,
                epoch,
                mollification_cone_angle_rad,
                cfg.mollification_sample_count,
            )
            training_tensor_decoder = tensorize_batch(data_to_dict(data_batch_decoder, active_feature_dim))

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
                sampler_opt=sampler_opt,
                sampler_scheduler=sampler_scheduler,
            )


            metrics = dict(train_metrics)

            active_validation_tensor = bootstrap_validation_tensor if phase == "bootstrap" else validation_tensor
            val_metrics = validate(model, active_validation_tensor, cfg, epoch, phase)
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
                mip_val_parts = [
                    f"{mip_level}:{metrics[f'brdf_val_loss_mip{mip_level}']:.4f}"
                    for mip_level in range(cfg.hierarchical_mip_count)
                    if f"brdf_val_loss_mip{mip_level}" in metrics
                ]
                mip_val_log = f" mip_val=[{' '.join(mip_val_parts)}]" if mip_val_parts else ""
                print(
                    f"[train] epoch {epoch:03d} "
                    f"phase={phase} brdf_loss={metrics['brdf_loss']:.6f}" \
                    f"{sampler_log} "
                    f"val_loss={metrics['brdf_val_loss']:.6f} "
                    f"{mip_val_log} "
                    f"yhat_mean={metrics['yhat_mean']:.3e} "
                    f"elapsed={elapsed:.1f}s"
                )

            if phase == "bootstrap" and metrics["brdf_val_loss"] < best_bootstrap_val_loss:
                best_bootstrap_val_loss = metrics["brdf_val_loss"]
                best_bootstrap_epoch = epoch
                best_bootstrap_metrics = dict(metrics)
                best_bootstrap_state = snapshot_model_state(model)
                print(f"[best-bootstrap] epoch {epoch:03d} val_loss={best_bootstrap_val_loss:.6f}")

            if phase == "finetune" and metrics["brdf_val_loss"] < best_finetune_val_loss:
                best_finetune_val_loss = metrics["brdf_val_loss"]
                best_finetune_epoch = epoch
                best_finetune_metrics = dict(metrics)
                best_finetune_state = snapshot_model_state(model)
                # Keep the sampler that was trained against this same decoder
                if cfg.train_importance_sampler and 'sampler_loss' in metrics:
                    best_sampler_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.importance_sampler.state_dict().items()
                    }
                    best_sampler_loss = metrics["sampler_loss"]
                print(f"[best-finetune] epoch {epoch:03d} val_loss={best_finetune_val_loss:.6f} and sampler_loss={best_sampler_loss:.6f}")

            if best_finetune_state is not None:
                best_brdf_val_loss = best_finetune_val_loss
                best_epoch = best_finetune_epoch
                best_phase = "finetune"
                best_metrics = best_finetune_metrics
                best_model_state = best_finetune_state
            elif best_bootstrap_state is not None:
                best_brdf_val_loss = best_bootstrap_val_loss
                best_epoch = best_bootstrap_epoch
                best_phase = "bootstrap"
                best_metrics = best_bootstrap_metrics
                best_model_state = best_bootstrap_state

            if run_logger.should_log_progress(
                epoch=epoch,
                phase_changed=phase_changed,
                is_final=(epoch == cfg.max_epochs - 1),
            ):
                run_logger.append_progress(epoch, metrics, phase)

            active_data_generator.release_data()
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

        if best_sampler_state is not None:
            model.importance_sampler.load_state_dict(best_sampler_state)

        best_ckpt_path = save_checkpoint(
            model,
            cfg,
            best_epoch,
            best_metrics,
            filename="best_checkpoint.pt",
        )
        print(
            f"[export] Restored best validation state from epoch "
            f"{best_epoch:03d} with val_loss={best_metrics.get('brdf_val_loss', float('nan')):.6f} and sampler_loss={best_sampler_loss:.6f}"
            f"and saved {best_ckpt_path}"
        )

    AssetConverter.export_renderer_assets(model, cfg)
    print("Done. Exports written to:", cfg.out_dir)


if __name__ == "__main__":
    main()
