#!/usr/bin/env python3
"""Create FP8-quantized latent runtime folders for quick render comparisons.

The renderer still consumes EXR textures. This script quantizes latent values to
an FP8-like value grid, then writes the quantized values back as half-float EXR.
This isolates value precision loss without requiring a native FP8 texture path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import Imath
import numpy as np
import OpenEXR


LATENT_PREFIXES = ("latent0", "latent1")


def read_exr_rgba(path: Path) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        arrays = []
        channels = header["channels"]
        for name in ("R", "G", "B", "A"):
            if name not in channels:
                raise ValueError(f"{path} is missing channel {name}")
            pixel_type = channels[name].type
            raw = exr.channel(name, pixel_type)
            if pixel_type == Imath.PixelType(Imath.PixelType.HALF):
                dtype = np.float16
            elif pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                dtype = np.float32
            else:
                raise ValueError(f"Unsupported EXR channel type in {path}: {pixel_type}")
            arrays.append(np.frombuffer(raw, dtype=dtype).reshape(height, width).astype(np.float32))
        return np.stack(arrays, axis=-1)
    finally:
        exr.close()


def write_exr_rgba_half(path: Path, rgba: np.ndarray) -> None:
    rgba = np.asarray(rgba, dtype=np.float16)
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError(f"Expected RGBA array, got shape {rgba.shape}")

    header = OpenEXR.Header(width, height)
    pt = Imath.PixelType(Imath.PixelType.HALF)
    header["channels"] = {name: Imath.Channel(pt) for name in ("R", "G", "B", "A")}
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels(
            {
                "R": rgba[:, :, 0].tobytes(),
                "G": rgba[:, :, 1].tobytes(),
                "B": rgba[:, :, 2].tobytes(),
                "A": rgba[:, :, 3].tobytes(),
            }
        )
    finally:
        out.close()


def latent_path(runtime_dir: Path, prefix: str, mip: int) -> Path:
    mip_path = runtime_dir / f"{prefix}_mip{mip}.exr"
    if mip_path.exists():
        return mip_path
    if mip == 0:
        fallback = runtime_dir / f"{prefix}.exr"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Missing {prefix} mip {mip} in {runtime_dir}")


def copy_non_latent_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            continue
        if item.name.startswith("latent0") or item.name.startswith("latent1"):
            continue
        shutil.copy2(item, dst / item.name)


def write_single_mip_metadata(src: Path, dst: Path, width: int, height: int, note: str) -> None:
    metadata_path = src / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mip_count"] = 1
    metadata["mip_shapes"] = [[int(width), int(height)]]
    metadata["width"] = int(width)
    metadata["height"] = int(height)
    metadata["latent_file_pattern"] = "latent{0,1}.exr"
    metadata["precision_test"] = note
    (dst / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def quantize_fp8_like(x: np.ndarray, mantissa_bits: int, emin: int, max_value: float) -> np.ndarray:
    """Round finite values to an FP8-like grid.

    This approximates FP8 storage by rounding each value to the nearest value
    available with a fixed number of mantissa bits for its exponent bin. Values
    too small for the normal range use a subnormal-like fixed step.
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out

    sign = np.sign(x[finite])
    ax = np.abs(x[finite])
    ax = np.minimum(ax, max_value)

    sub_step = np.float32(2.0 ** (emin - mantissa_bits))
    normal = ax >= np.float32(2.0**emin)

    q = np.empty_like(ax, dtype=np.float32)
    q[~normal] = np.round(ax[~normal] / sub_step) * sub_step
    if np.any(normal):
        exp = np.floor(np.log2(ax[normal])).astype(np.float32)
        step = np.exp2(exp - mantissa_bits)
        q[normal] = np.round(ax[normal] / step) * step

    q = np.minimum(q, max_value)
    out[finite] = sign * q
    return out


def quantize(rgba: np.ndarray, mode: str) -> np.ndarray:
    if mode == "e4m3":
        return quantize_fp8_like(rgba, mantissa_bits=3, emin=-6, max_value=448.0)
    if mode == "e5m2":
        return quantize_fp8_like(rgba, mantissa_bits=2, emin=-14, max_value=57344.0)
    raise ValueError(f"Unsupported mode: {mode}")


def stats(original: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    diff = quantized.astype(np.float32) - original.astype(np.float32)
    return {
        "min": float(np.min(original)),
        "max": float(np.max(original)),
        "mean": float(np.mean(original)),
        "std": float(np.std(original)),
        "quantized_min": float(np.min(quantized)),
        "quantized_max": float(np.max(quantized)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "zero_fraction": float(np.mean(quantized == 0.0)),
    }


def prepare(src: Path, out_root: Path, mode: str) -> None:
    baseline = out_root / "mip0_baseline_runtime"
    quantized_dir = out_root / f"mip0_fp8_{mode}_runtime"

    for dst in (baseline, quantized_dir):
        if dst.exists():
            shutil.rmtree(dst)
        copy_non_latent_files(src, dst)

    all_stats: dict[str, dict[str, float]] = {}
    width = height = 0
    for prefix in LATENT_PREFIXES:
        rgba = read_exr_rgba(latent_path(src, prefix, 0))
        height, width = rgba.shape[:2]
        q = quantize(rgba, mode)

        write_exr_rgba_half(baseline / f"{prefix}.exr", rgba)
        write_exr_rgba_half(quantized_dir / f"{prefix}.exr", q)
        all_stats[prefix] = stats(rgba, q)

    write_single_mip_metadata(src, baseline, width, height, "mip0 baseline copied from source")
    write_single_mip_metadata(src, quantized_dir, width, height, f"mip0 quantized to fp8-like {mode} grid")

    stats_path = out_root / f"fp8_{mode}_quantization_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"[fp8] baseline:  {baseline}")
    print(f"[fp8] quantized: {quantized_dir}")
    print(f"[fp8] stats:     {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mip0 baseline and FP8-like quantized latent runtime folders.")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("e4m3", "e5m2"), default="e4m3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(args.source_runtime.resolve(), args.out_root.resolve(), args.mode)


if __name__ == "__main__":
    main()
