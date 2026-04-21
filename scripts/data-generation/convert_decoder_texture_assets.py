#!/usr/bin/env python3
import argparse
import json
import os
import struct
import re
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


def infer_decoder_layout(weights: dict):
    frame_weight = np.asarray(weights["frame_linear.weight"], dtype=np.float32)
    if frame_weight.ndim != 2:
        raise ValueError(f"frame_linear.weight must be 2D, got {frame_weight.shape}")

    latent_ch = int(np.asarray(weights.get("latent_ch", np.array([frame_weight.shape[1]], dtype=np.int32))).reshape(-1)[0])
    num_frames = int(np.asarray(weights.get("num_frames", np.array([frame_weight.shape[0] // 6], dtype=np.int32))).reshape(-1)[0])
    exp_offset = float(np.asarray(weights.get("exp_offset", np.array([3.0], dtype=np.float32))).reshape(-1)[0])

    mlp_weight_names = []
    for key in weights.keys():
        match = re.fullmatch(r"mlp\.(\d+)\.weight", key)
        if match:
            mlp_weight_names.append((int(match.group(1)), key))
    mlp_weight_names.sort()

    if not mlp_weight_names:
        raise ValueError("No MLP weights found in decoder state.")

    linear_layers = [
        {
            "name": "frame_linear",
            "weight_name": "frame_linear.weight",
            "bias_name": "frame_linear.bias" if "frame_linear.bias" in weights else None,
            "weight_shape": tuple(int(x) for x in frame_weight.shape),
        }
    ]

    for layer_idx, weight_name in mlp_weight_names:
        weight = np.asarray(weights[weight_name], dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"{weight_name} must be 2D, got {weight.shape}")
        bias_name = f"mlp.{layer_idx}.bias"
        linear_layers.append(
            {
                "name": f"mlp.{layer_idx}",
                "weight_name": weight_name,
                "bias_name": bias_name if bias_name in weights else None,
                "weight_shape": tuple(int(x) for x in weight.shape),
            }
        )

    mlp_width = int(linear_layers[1]["weight_shape"][0]) if len(linear_layers) > 1 else 0
    mlp_depth = max(0, len(linear_layers) - 2)  # exclude frame_linear and output layer

    decoder_layout = {}
    for layer in linear_layers:
        decoder_layout[layer["weight_name"]] = list(layer["weight_shape"])
        if layer["bias_name"] is not None:
            bias = np.asarray(weights[layer["bias_name"]], dtype=np.float32)
            decoder_layout[layer["bias_name"]] = list(bias.shape)

    return {
        "latent_ch": latent_ch,
        "num_frames": num_frames,
        "exp_offset": exp_offset,
        "linear_layers": linear_layers,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "decoder_layout": decoder_layout,
    }


def is_legacy_2x32_layout(layout: dict) -> bool:
    expected = {
        "frame_linear.weight": [12, 8],
        "mlp.0.weight": [32, 20],
        "mlp.0.bias": [32],
        "mlp.2.weight": [32, 32],
        "mlp.2.bias": [32],
        "mlp.4.weight": [3, 32],
        "mlp.4.bias": [3],
    }
    return layout["decoder_layout"] == expected


def get_supported_runtime_layout(layout: dict):
    if layout["latent_ch"] != 8:
        return None
    if layout["num_frames"] != 2:
        return None
    if layout["decoder_layout"].get("frame_linear.weight") != [12, 8]:
        return None
    if "frame_linear.bias" in layout["decoder_layout"]:
        return None

    mlp_layers = layout["linear_layers"][1:]
    if len(mlp_layers) < 2:
        return None

    mlp_depth = len(mlp_layers) - 1
    if mlp_depth not in (2, 3):
        return None

    first_hidden_shape = mlp_layers[0]["weight_shape"]
    if len(first_hidden_shape) != 2:
        return None

    mlp_width = int(first_hidden_shape[0])
    if mlp_width not in (16, 32, 64):
        return None
    if int(first_hidden_shape[1]) != 20:
        return None

    for hidden_index, hidden_layer in enumerate(mlp_layers[:-1]):
        rows, cols = hidden_layer["weight_shape"]
        if hidden_layer["bias_name"] is None:
            return None
        expected_cols = 20 if hidden_index == 0 else mlp_width
        if int(rows) != mlp_width or int(cols) != expected_cols:
            return None
        bias_shape = layout["decoder_layout"].get(hidden_layer["bias_name"])
        if bias_shape != [mlp_width]:
            return None

    output_layer = mlp_layers[-1]
    if output_layer["bias_name"] is None:
        return None
    if list(output_layer["weight_shape"]) != [3, mlp_width]:
        return None
    if layout["decoder_layout"].get(output_layer["bias_name"]) != [3]:
        return None

    return {
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
    }

def save_weights_bin(path: Path, weights: dict):
    """
    Binary layout (little endian):

    Legacy format (NMDLWT01) is kept for the original 2x32 network.
    Family format (NMDLWT02) supports 16x2, 32x2, 64x2, and 64x3.
    """
    layout = infer_decoder_layout(weights)
    latent_ch = layout["latent_ch"]
    num_frames = layout["num_frames"]
    exp_offset = layout["exp_offset"]
    runtime_layout = get_supported_runtime_layout(layout)

    with open(path, "wb") as f:
        if is_legacy_2x32_layout(layout):
            ordered = [
                ("frame_linear.weight", (12, 8)),
                ("mlp.0.weight", (32, 20)),
                ("mlp.0.bias", (32,)),
                ("mlp.2.weight", (32, 32)),
                ("mlp.2.bias", (32,)),
                ("mlp.4.weight", (3, 32)),
                ("mlp.4.bias", (3,)),
            ]

            f.write(b"NMDLWT01")
            f.write(struct.pack("<iii f", latent_ch, num_frames, 1, exp_offset))

            for name, expected_shape in ordered:
                arr = np.asarray(weights[name], dtype=np.float32)
                if tuple(arr.shape) != expected_shape:
                    raise ValueError(f"{name} expected shape {expected_shape}, got {arr.shape}")
                f.write(arr.astype(np.float32).ravel(order="C").tobytes())
            return

        if runtime_layout is None:
            raise ValueError(
                "Unsupported decoder layout for runtime export. "
                "Supported layouts are 16x2, 32x2, 64x2, and 64x3."
            )

        f.write(b"NMDLWT02")
        f.write(
            struct.pack(
                "<iii f ii",
                latent_ch,
                num_frames,
                1,
                exp_offset,
                runtime_layout["mlp_width"],
                runtime_layout["mlp_depth"],
            )
        )

        ordered_layers = ["frame_linear"]
        ordered_layers.extend(f"mlp.{2 * i}" for i in range(runtime_layout["mlp_depth"] + 1))

        for layer_name in ordered_layers:
            weight_name = f"{layer_name}.weight"
            bias_name = f"{layer_name}.bias"
            weight = np.asarray(weights[weight_name], dtype=np.float32)
            f.write(weight.ravel(order="C").tobytes())
            if bias_name in weights:
                bias = np.asarray(weights[bias_name], dtype=np.float32)
                f.write(bias.ravel(order="C").tobytes())

def save_metadata(path: Path, latent: np.ndarray, weights: dict):
    _, h, w = latent.shape
    layout = infer_decoder_layout(weights)
    runtime_layout = get_supported_runtime_layout(layout)
    metadata = {
        "width": int(w),
        "height": int(h),
        "latent_dim": int(latent.shape[0]),
        "num_frames": layout["num_frames"],
        "exp_offset": layout["exp_offset"],
        "apply_exp": True,
        "mlp_width": layout["mlp_width"],
        "mlp_depth": layout["mlp_depth"],
        "weight_file_format": "NMDLWT01" if is_legacy_2x32_layout(layout) else ("NMDLWT02" if runtime_layout is not None else "unsupported"),
        "decoder_layout": layout["decoder_layout"],
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
