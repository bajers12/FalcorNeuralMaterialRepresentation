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
  If you pass --use_normals, the loader must provide 'normal' and wi/wo are assumed
  to be in the SAME space as normal (typically world space). The script will build a
  local frame from 'normal' and transform wi/wo into local space so that the decoder
  sees directions with local +Z aligned to the normal.

Exports:
  - latent_texture.pt: {"Z": [1,C,H,W], "shape": (H,W,C)}
  - latent_rgba0.npz / latent_rgba1.npz if C==8 (for renderer-friendly RGBA splits)
  - decoder.pt: PyTorch state_dict
  - decoder_weights.npz: Numpy arrays for renderer-side loading
"""

from __future__ import annotations

import os
import math
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
from DataGenerator import DataGenerator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrainConfig:
    # Data
    train_path: str
    val_path: Optional[str] = None
    num_workers: int = 4

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
    log_eps: float = 1e-6          # y' = clamp(y, eps) for log

    # Optimization
    device: str = "cuda"
    seed: int = 1337
    batch_size: int = 65536
    max_epochs: int = 50

    lr: float = 1e-3
    lr_min: float = 1e-4
    lr_latent: Optional[float] = None
    lr_decoder: Optional[float] = None
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None

    # Logging / checkpoints
    out_dir: str = "./out_neural_material_mvp"
    save_every: int = 5
    print_every_steps: int = 50

    # Training behavior
    train_latent_texture: bool = True
    train_decoder: bool = True
    freeze_latent_after_epoch: Optional[int] = None
    freeze_decoder_after_epoch: Optional[int] = None
    freeze_latent_after_step: Optional[int] = None
    freeze_decoder_after_step: Optional[int] = None

    # Normals support
    use_normals: bool = False  # if True, expects batch['normal'] and converts wi/wo -> local


# =============================================================================
# Dataset
# =============================================================================

class StreamingDataset(Dataset):
    def __init__(self, batchsize: int):
        self.uv = None
        self.wi = None
        self.wo = None
        self.y  = None
        self.batchsize = batchsize

    def update(self, data_dict):
        self.uv = torch.from_numpy(data_dict["uv"]).float()
        self.wi = torch.from_numpy(data_dict["wi"]).float()
        self.wo = torch.from_numpy(data_dict["wo"]).float()
        self.y  = torch.from_numpy(data_dict["y"]).float()

    def __len__(self):
        return self.batchsize

    def __getitem__(self, idx):
        return {
            "uv": self.uv[idx],
            "wi": self.wi[idx],
            "wo": self.wo[idx],
            "y": self.y[idx],
        }

def make_dataloader(ds: Dataset, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    """
    Creates a DataLoader. pin_memory is only useful when training on CUDA.
    """
    use_pin = (cfg.device == "cuda")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=shuffle,
        persistent_workers=(cfg.num_workers > 0),
    )


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
            self.Z,                                # [1,C,H,W]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )                                          # [1,C,B,1]
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

    def _predict_frames(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z [B,C] -> T,B,N each [B,F,3]
        """
        Bsz = z.shape[0]
        ft = self.frame_linear(z).view(Bsz, self.num_frames, 6)
        N = self._safe_normalize(ft[..., 0:3])
        T = self._safe_normalize(ft[..., 3:6])
        Bv = self._safe_normalize(torch.cross(N, T, dim=-1))
        return T, Bv, N

    def forward_raw(self, z: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
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

        dir_feats = torch.cat([wi_f, wo_f], dim=-1).view(z.shape[0], 6 * self.num_frames)
        x = torch.cat([z, dir_feats], dim=-1)
        return self.mlp(x)

    def forward(self, z: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(z, wi, wo)
        return torch.exp(raw - self.exp_offset)


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

    def forward_with_raw(self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.latent.sample(uv)
        raw = self.decoder.forward_raw(z, wi, wo)
        y_hat = torch.exp(raw - self.decoder.exp_offset)
        return y_hat, raw

    def forward(self, uv: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        y_hat, _raw = self.forward_with_raw(uv, wi, wo)
        return y_hat


# =============================================================================
# Normals -> local frame helpers
# =============================================================================

def build_tbn(n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a stable tangent frame from per-sample normals.

    n: [B,3] (not necessarily perfectly normalized)
    returns (t, b, n_unit): each [B,3]
    """
    n_unit = n / n.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # Choose an "up" that is not parallel to n.
    up = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n_unit)
    alt = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n_unit)
    use_alt = (n_unit[:, 1].abs() > 0.99).unsqueeze(1)
    up = torch.where(use_alt, alt, up)

    t = torch.cross(up, n_unit, dim=1)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b = torch.cross(n_unit, t, dim=1)
    return t, b, n_unit


def to_local(v_world: torch.Tensor, t: torch.Tensor, b: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
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

def log_l1_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    L1 loss in log space:
      mean(|log(y_hat+eps) - log(y+eps)|)
    """
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


def make_optimizer(model: NeuralMaterialModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    param_groups = []
    if cfg.train_latent_texture:
        latent_params = [p for p in model.latent.parameters() if p.requires_grad]
        if latent_params:
            param_groups.append({"params": latent_params, "lr": _latent_lr(cfg), "name": "latent"})
    if cfg.train_decoder:
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": _decoder_lr(cfg), "name": "decoder"})
    if not param_groups:
        raise ValueError("Nothing to train: both train_latent_texture and train_decoder are False or all params are frozen")
    return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)


def make_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    """
    Cosine LR decay from the per-group base LR to the per-group minimum over cfg.max_epochs (epoch-stepped).
    """
    base_by_name = {"latent": _latent_lr(cfg), "decoder": _decoder_lr(cfg)}
    min_by_name = {"latent": _latent_lr_min(cfg), "decoder": _decoder_lr_min(cfg)}

    def lr_lambda_factory(group_name: str):
        base = base_by_name[group_name]
        min_lr = min_by_name[group_name]

        def lr_lambda(epoch: int):
            if cfg.max_epochs <= 1:
                return min_lr / max(base, 1e-12)
            t = epoch / (cfg.max_epochs - 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))
            lr_now = min_lr + (base - min_lr) * scale
            return lr_now / max(base, 1e-12)

        return lr_lambda

    lambdas = [lr_lambda_factory(pg.get("name", "latent")) for pg in opt.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(opt, lambdas)


def maybe_freeze_parts(model: NeuralMaterialModel, cfg: TrainConfig, *, epoch: Optional[int] = None, global_step: Optional[int] = None) -> None:
    if epoch is not None:
        if cfg.freeze_latent_after_epoch is not None and epoch >= cfg.freeze_latent_after_epoch:
            for p in model.latent.parameters():
                p.requires_grad_(False)
        if cfg.freeze_decoder_after_epoch is not None and epoch >= cfg.freeze_decoder_after_epoch:
            for p in model.decoder.parameters():
                p.requires_grad_(False)

    if global_step is not None:
        if cfg.freeze_latent_after_step is not None and global_step >= cfg.freeze_latent_after_step:
            for p in model.latent.parameters():
                p.requires_grad_(False)
        if cfg.freeze_decoder_after_step is not None and global_step >= cfg.freeze_decoder_after_step:
            for p in model.decoder.parameters():
                p.requires_grad_(False)


def maybe_rebuild_optimizer_and_scheduler(model: NeuralMaterialModel, opt: torch.optim.Optimizer, scheduler, cfg: TrainConfig):
    active_group_names = []
    if cfg.train_latent_texture and any(p.requires_grad for p in model.latent.parameters()):
        active_group_names.append("latent")
    if cfg.train_decoder and any(p.requires_grad for p in model.decoder.parameters()):
        active_group_names.append("decoder")

    current_group_names = [pg.get("name") for pg in opt.param_groups]
    if active_group_names != current_group_names:
        new_opt = make_optimizer(model, cfg)
        new_scheduler = make_scheduler(new_opt, cfg)
        if scheduler is not None and hasattr(scheduler, "last_epoch"):
            new_scheduler.last_epoch = scheduler.last_epoch
        print(f"[train] rebuilt optimizer groups: {current_group_names} -> {active_group_names}")
        return new_opt, new_scheduler

    return opt, scheduler


def _maybe_transform_dirs_with_normals(batch: Dict[str, torch.Tensor], cfg: TrainConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If cfg.use_normals:
      - expects batch['normal']
      - converts wi/wo from that space into local space aligned with normal (+Z)
    Else:
      - returns wi/wo as-is
    """
    wi = batch["wi"].to(device, non_blocking=True)
    wo = batch["wo"].to(device, non_blocking=True)

    if not cfg.use_normals:
        return wi, wo

    if "normal" not in batch:
        raise ValueError("You passed --use_normals but your dataset batch has no 'normal' key.")

    n = batch["normal"].to(device, non_blocking=True)
    t, b, n_unit = build_tbn(n)
    wi_l = to_local(wi, t, b, n_unit)
    wo_l = to_local(wo, t, b, n_unit)
    return wi_l, wo_l


def train_one_epoch(
    model: NeuralMaterialModel,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    epoch: int,
    global_step_start: int = 0,
):
    model.train()
    device = torch.device(cfg.device)

    t0 = time.time()
    running_loss = 0.0
    running_mae = 0.0
    running_yhat_mean = 0.0
    running_y_mean = 0.0
    running_raw_mean = 0.0
    running_raw_std = 0.0
    n_batches = 0
    global_step = global_step_start
    decoder_frozen_logged = False
    latent_frozen_logged = False

    for step, batch in enumerate(loader):
        maybe_freeze_parts(model, cfg, global_step=global_step)
        opt, scheduler = maybe_rebuild_optimizer_and_scheduler(model, opt, scheduler, cfg)

        decoder_now_frozen = all(not p.requires_grad for p in model.decoder.parameters())
        if decoder_now_frozen and not decoder_frozen_logged:
            print(f"[train] freezing decoder at global_step={global_step}")
            decoder_frozen_logged = True

        latent_now_frozen = all(not p.requires_grad for p in model.latent.parameters())
        if latent_now_frozen and not latent_frozen_logged:
            print(f"[train] freezing latent texture at global_step={global_step}")
            latent_frozen_logged = True

        uv = batch["uv"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        wi, wo = _maybe_transform_dirs_with_normals(batch, cfg, device)

        if cfg.clamp_min_target > 0.0:
            y = y.clamp_min(cfg.clamp_min_target)

        y_hat, raw = model.forward_with_raw(uv, wi, wo)
        loss = log_l1_loss(y_hat, y, cfg.log_eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        opt.step()

        with torch.no_grad():
            stats = compute_basic_stats(y_hat, y)
            raw_stats = compute_raw_stats(raw)

        running_loss += loss.item()
        running_mae += stats["mae"]
        running_yhat_mean += stats["yhat_mean"]
        running_y_mean += stats["y_mean"]
        running_raw_mean += raw_stats["raw_mean"]
        running_raw_std += raw_stats["raw_std"]
        n_batches += 1
        global_step += 1

        if (step + 1) % cfg.print_every_steps == 0:
            dt = time.time() - t0
            lr_summary = ", ".join([f"{pg.get('name','group')}={pg['lr']:.2e}" for pg in opt.param_groups])
            print(
                f"[train] epoch {epoch:03d} step {step+1:05d} global_step={global_step:07d} "
                f"loss={running_loss/n_batches:.6f} mae={running_mae/n_batches:.6f} "
                f"yhat_mean={running_yhat_mean/n_batches:.3e} y_mean={running_y_mean/n_batches:.3e} "
                f"raw_mean={running_raw_mean/n_batches:.3f} raw_std={running_raw_std/n_batches:.3f} "
                f"lr[{lr_summary}] time={dt:.1f}s"
            )

    return {
        "loss": running_loss / max(1, n_batches),
        "mae": running_mae / max(1, n_batches),
        "yhat_mean": running_yhat_mean / max(1, n_batches),
        "y_mean": running_y_mean / max(1, n_batches),
        "raw_mean": running_raw_mean / max(1, n_batches),
        "raw_std": running_raw_std / max(1, n_batches),
    }, global_step, opt, scheduler


@torch.no_grad()
def validate(model: NeuralMaterialModel, loader: DataLoader, cfg: TrainConfig, epoch: int) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)

    running_loss = 0.0
    running_mae = 0.0
    running_yhat_mean = 0.0
    running_y_mean = 0.0
    running_raw_mean = 0.0
    running_raw_std = 0.0
    n_batches = 0

    for batch in loader:
        uv = batch["uv"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        wi, wo = _maybe_transform_dirs_with_normals(batch, cfg, device)

        if cfg.clamp_min_target > 0.0:
            y = y.clamp_min(cfg.clamp_min_target)

        y_hat, raw = model.forward_with_raw(uv, wi, wo)
        loss = log_l1_loss(y_hat, y, cfg.log_eps)
        stats = compute_basic_stats(y_hat, y)
        raw_stats = compute_raw_stats(raw)

        running_loss += loss.item()
        running_mae += stats["mae"]
        running_yhat_mean += stats["yhat_mean"]
        running_y_mean += stats["y_mean"]
        running_raw_mean += raw_stats["raw_mean"]
        running_raw_std += raw_stats["raw_std"]
        n_batches += 1

    out = {
        "val_loss": running_loss / max(1, n_batches),
        "val_mae": running_mae / max(1, n_batches),
        "val_yhat_mean": running_yhat_mean / max(1, n_batches),
        "val_y_mean": running_y_mean / max(1, n_batches),
        "val_raw_mean": running_raw_mean / max(1, n_batches),
        "val_raw_std": running_raw_std / max(1, n_batches),
    }
    print(
        f"[val] epoch {epoch:03d} val_loss={out['val_loss']:.6f} val_mae={out['val_mae']:.6f} "
        f"val_yhat_mean={out['val_yhat_mean']:.3e} val_y_mean={out['val_y_mean']:.3e} "
        f"val_raw_mean={out['val_raw_mean']:.3f} val_raw_std={out['val_raw_std']:.3f}"
    )
    return out


# =============================================================================
# Export / Checkpoints
# =============================================================================

def save_checkpoint(model: NeuralMaterialModel, opt, scheduler, cfg: TrainConfig, epoch: int, metrics: Dict[str, float]) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    payload = {
        "epoch": epoch,
        "config": asdict(cfg),
        "metrics": metrics,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_model_weights_from_checkpoint(model: NeuralMaterialModel, ckpt_path: str, device: torch.device) -> Dict:
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    return payload



def export_latent_texture(model: NeuralMaterialModel, cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    Z = model.latent.Z.detach().cpu()  # [1,C,H,W]
    torch.save({"Z": Z, "shape": (cfg.tex_h, cfg.tex_w, cfg.latent_ch)}, os.path.join(cfg.out_dir, "latent_texture.pt"))

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


def save_config(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)

    p.add_argument("--out_dir", type=str, default="./out_neural_material_mvp")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--tex_h", type=int, default=512)
    p.add_argument("--tex_w", type=int, default=512)
    p.add_argument("--latent_ch", type=int, default=8)

    p.add_argument("--num_frames", type=int, default=2)
    p.add_argument("--mlp_width", type=int, default=32)
    p.add_argument("--mlp_depth", type=int, default=2)
    p.add_argument("--exp_offset", type=float, default=3.0)

    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--lr_latent", type=float, default=None)
    p.add_argument("--lr_decoder", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=None)

    p.add_argument("--log_eps", type=float, default=1e-6)
    p.add_argument("--clamp_min_target", type=float, default=0.0)

    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--print_every_steps", type=int, default=50)

    p.add_argument("--train_latent_texture", action="store_true")
    p.add_argument("--no_train_latent_texture", action="store_true")
    p.add_argument("--train_decoder", action="store_true")
    p.add_argument("--no_train_decoder", action="store_true")

    p.add_argument("--freeze_latent_after_epoch", type=int, default=None)
    p.add_argument("--freeze_decoder_after_epoch", type=int, default=None)
    p.add_argument("--freeze_latent_after_step", type=int, default=None)
    p.add_argument("--freeze_decoder_after_step", type=int, default=None)

    # normals flag
    p.add_argument("--use_normals", action="store_true", help="Use per-sample normals to transform wi/wo into local frame.")

    args = p.parse_args()

    cfg = TrainConfig(train_path=args.train_path)
    cfg.val_path = args.val_path
    cfg.out_dir = args.out_dir
    cfg.device = args.device
    cfg.seed = args.seed

    cfg.tex_h = args.tex_h
    cfg.tex_w = args.tex_w
    cfg.latent_ch = args.latent_ch

    cfg.num_frames = args.num_frames
    cfg.mlp_width = args.mlp_width
    cfg.mlp_depth = args.mlp_depth
    cfg.exp_offset = args.exp_offset

    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.lr = args.lr
    cfg.lr_min = args.lr_min
    cfg.lr_latent = args.lr_latent
    cfg.lr_decoder = args.lr_decoder
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip_norm = args.grad_clip_norm

    cfg.log_eps = args.log_eps
    cfg.clamp_min_target = args.clamp_min_target

    cfg.num_workers = args.num_workers
    cfg.save_every = args.save_every
    cfg.print_every_steps = args.print_every_steps

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
    cfg.freeze_latent_after_step = args.freeze_latent_after_step
    cfg.freeze_decoder_after_step = args.freeze_decoder_after_step

    cfg.use_normals = args.use_normals

    return cfg

def generate_new_data(data_generator: DataGenerator):
    data = data_generator.generate_data(0)

    # unpack (must match your struct layout!)
    uv = data[:, 0:2]
    wo = data[:, 2:5]
    wi = data[:, 5:8]
    f  = data[:, 8:11]

    # your training expects y = f * cos term already baked
    y = f

    return {
        "uv": uv,
        "wi": wi,
        "wo": wo,
        "y": y,
    }

# =============================================================================
# Main
# =============================================================================

def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    data_generator = DataGenerator(sampleCount=cfg.batch_size)

    # Device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        cfg.device = "cpu"

    device = torch.device(cfg.device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_config(cfg)
    print("Config:", json.dumps(asdict(cfg), indent=2))

    # Data
    train_ds = StreamingDataset(batchsize=cfg.batch_size)
    train_loader = make_dataloader(train_ds, cfg, shuffle=True)

    val_loader = None
    if cfg.val_path:
        val_ds = StreamingDataset(batchsize=cfg.batch_size)
        val_loader = make_dataloader(val_ds, cfg, shuffle=False)

    # Model
    model = NeuralMaterialModel(cfg).to(device)
    opt = make_optimizer(model, cfg)
    scheduler = make_scheduler(opt, cfg)

    best_val = float("inf")
    best_ckpt_path: Optional[str] = None
    best_epoch: Optional[int] = None
    global_step = 0

    for epoch in range(cfg.max_epochs):
        maybe_freeze_parts(model, cfg, epoch=epoch, global_step=global_step)

        data_batch = generate_new_data(data_generator)
        train_ds.update(data_batch)

        train_metrics, global_step, opt, scheduler = train_one_epoch(
            model, train_loader, opt, scheduler, cfg, epoch, global_step_start=global_step
        )

        # Step scheduler AFTER training epoch
        scheduler.step()

        metrics = dict(train_metrics)
        metrics["global_step"] = global_step

        if val_loader is not None:
            val_metrics = validate(model, val_loader, cfg, epoch)
            metrics.update(val_metrics)
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                best_epoch = epoch
                best_ckpt_path = save_checkpoint(model, opt, scheduler, cfg, epoch, metrics)
                print(f"[best] epoch {epoch:03d} val_loss={best_val:.6f} checkpoint={best_ckpt_path}")
        else:
            if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.max_epochs - 1:
                save_checkpoint(model, opt, scheduler, cfg, epoch, metrics)

        data_generator.release_data()

    if val_loader is not None and best_ckpt_path is not None:
        payload = load_model_weights_from_checkpoint(model, best_ckpt_path, device)
        print(
            f"[export] Reloaded best validation checkpoint from epoch "
            f"{payload.get('epoch', best_epoch):03d} with val_loss="
            f"{payload.get('metrics', {}).get('val_loss', float('nan')):.6f}"
        )

    export_latent_texture(model, cfg)
    export_decoder_weights(model, cfg)
    print("Done. Exports written to:", cfg.out_dir)


if __name__ == "__main__":
    main()
