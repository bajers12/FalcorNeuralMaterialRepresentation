#!/usr/bin/env python3
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    raise RuntimeError("This script needs PyTorch installed.")

# EXR writing: tries OpenEXR first, then imageio as fallback.
def write_exr(path: Path, rgba_hw4: np.ndarray):
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
        exr.writePixels({
            "R": rgba_hw4[:, :, 0].astype(np.float32).tobytes(),
            "G": rgba_hw4[:, :, 1].astype(np.float32).tobytes(),
            "B": rgba_hw4[:, :, 2].astype(np.float32).tobytes(),
            "A": rgba_hw4[:, :, 3].astype(np.float32).tobytes(),
        })
        exr.close()
        return
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        iio.imwrite(str(path), rgba_hw4)
        return
    except Exception as e:
        raise RuntimeError(
            f"Failed to write EXR '{path}'. Install either OpenEXR or imageio with EXR support.\n{e}"
        )

def load_latents(latent_texture_pt, latent_rgba0_npz, latent_rgba1_npz):
    latent_texture_pt = Path(latent_texture_pt)
    latent_rgba0_npz = Path(latent_rgba0_npz)
    latent_rgba1_npz = Path(latent_rgba1_npz)

    if latent_rgba0_npz.exists() and latent_rgba1_npz.exists():
        z0 = np.load(latent_rgba0_npz)["rgba"].astype(np.float32)   # 4,H,W
        z1 = np.load(latent_rgba1_npz)["rgba"].astype(np.float32)   # 4,H,W
        assert z0.shape[0] == 4 and z1.shape[0] == 4
        latent = np.concatenate([z0, z1], axis=0)                   # 8,H,W
    elif latent_texture_pt.exists():
        obj = torch.load(latent_texture_pt, map_location="cpu")
        latent = obj["Z"][0].detach().cpu().numpy().astype(np.float32)  # 8,H,W
    else:
        raise FileNotFoundError("Need either latent_rgba0/1.npz or latent_texture.pt")

    assert latent.shape[0] == 8, f"Expected latent channels=8, got {latent.shape}"
    return latent

def load_weights(decoder_pt, decoder_weights_npz):
    decoder_pt = Path(decoder_pt)
    decoder_weights_npz = Path(decoder_weights_npz)

    weights = {}
    if decoder_weights_npz.exists():
        npz = np.load(decoder_weights_npz)
        for k in npz.files:
            weights[k] = npz[k]
    elif decoder_pt.exists():
        sd = torch.load(decoder_pt, map_location="cpu")
        for k, v in sd.items():
            weights[k] = v.detach().cpu().numpy()
        weights["latent_ch"] = np.array([8], dtype=np.int32)
        weights["num_frames"] = np.array([2], dtype=np.int32)
        weights["exp_offset"] = np.array([3.0], dtype=np.float32)
    else:
        raise FileNotFoundError("Need either decoder_weights.npz or decoder.pt")

    return weights

def save_weights_bin(path: Path, weights: dict):
    """
    Binary layout (little endian):

    magic[8]      = b'NMDLWT01'
    int32 latent_ch
    int32 num_frames
    int32 apply_exp   (1 for now)
    float exp_offset

    Then raw arrays in this exact order, all float32 row-major:
      frame_linear.weight   shape (12, 8)
      mlp.0.weight          shape (32, 20)
      mlp.0.bias            shape (32,)
      mlp.2.weight          shape (32, 32)
      mlp.2.bias            shape (32,)
      mlp.4.weight          shape (3, 32)
      mlp.4.bias            shape (3,)
    """
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
        f.write(struct.pack("<iii f", latent_ch, num_frames, 1, exp_offset))

        for name, expected_shape in ordered:
            arr = np.asarray(weights[name], dtype=np.float32)
            if tuple(arr.shape) != expected_shape:
                raise ValueError(f"{name} expected shape {expected_shape}, got {arr.shape}")
            f.write(arr.astype(np.float32).ravel(order="C").tobytes())

def save_metadata(path: Path, latent: np.ndarray, weights: dict):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-pt", default="decoder.pt")
    ap.add_argument("--decoder-weights-npz", default="decoder_weights.npz")
    ap.add_argument("--latent-texture-pt", default="latent_texture.pt")
    ap.add_argument("--latent-rgba0-npz", default="latent_rgba0.npz")
    ap.add_argument("--latent-rgba1-npz", default="latent_rgba1.npz")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent = load_latents(args.latent_texture_pt, args.latent_rgba0_npz, args.latent_rgba1_npz)
    weights = load_weights(args.decoder_pt, args.decoder_weights_npz)

    # latent is 8,H,W -> split to two H,W,4 textures
    z0 = latent[0:4].transpose(1, 2, 0).copy()  # H,W,4
    z1 = latent[4:8].transpose(1, 2, 0).copy()  # H,W,4

    write_exr(out_dir / "latent0.exr", z0)
    write_exr(out_dir / "latent1.exr", z1)

    # Save debug copies too.
    np.save(out_dir / "latent0.npy", z0)
    np.save(out_dir / "latent1.npy", z1)

    save_weights_bin(out_dir / "decoder_weights.bin", weights)
    save_metadata(out_dir / "metadata.json", latent, weights)

    print(f"Wrote assets to: {out_dir}")
    print("  latent0.exr")
    print("  latent1.exr")
    print("  decoder_weights.bin")
    print("  metadata.json")
    print("  latent0.npy")
    print("  latent1.npy")

if __name__ == "__main__":
    main()