#!/usr/bin/env python3
"""Export renderer-ready NeuralMaterial assets from an OnlineStepfreeze checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = REPO_ROOT / "build" / "windows-vs2022" / "bin" / "Release"
PLUGIN_DIR = BIN_DIR / "plugins"
PYTHON_DIR = BIN_DIR / "python"

if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))
if os.name == "nt":
    for dll_dir in (BIN_DIR, PLUGIN_DIR):
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))
path_parts = [str(path) for path in (BIN_DIR, PLUGIN_DIR) if path.exists()]
if path_parts:
    os.environ["PATH"] = os.pathsep.join(path_parts + [os.environ.get("PATH", "")])

import AssetConverter
from OnlineStepfreeze import NeuralMaterialModel, TrainConfig


def load_config(config_dict: dict) -> TrainConfig:
    cfg = TrainConfig()
    valid_names = {field.name for field in fields(TrainConfig)}
    for key, value in config_dict.items():
        if key in valid_names:
            setattr(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load best_checkpoint.pt and re-export EXR/weight/metadata runtime assets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preview_out_dir", type=Path, required=True)
    parser.add_argument(
        "--export_numpy_debug",
        action="store_true",
        help="Also write large latent*.npy debug dumps. Runtime does not need these.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_config(checkpoint["config"])
    cfg.preview_out_dir = str(args.preview_out_dir.resolve())
    cfg.export_numpy_debug = bool(args.export_numpy_debug)

    model = NeuralMaterialModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    AssetConverter.export_renderer_assets(
        model,
        cfg,
        write_numpy_debug=cfg.export_numpy_debug,
    )


if __name__ == "__main__":
    main()
