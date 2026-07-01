from __future__ import annotations

import numpy as np
from pathlib import Path
import struct
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from OnlineStepfreeze import NeuralMaterialModel, TrainConfig

def write_exr(path: Path, rgba_hw4: np.ndarray) -> None:
    rgba_hw4 = np.asarray(rgba_hw4, dtype=np.float16)
    assert rgba_hw4.ndim == 3 and rgba_hw4.shape[2] == 4, f"Expected HxWx4, got {rgba_hw4.shape}"

    h, w, _ = rgba_hw4.shape

    try:
        import OpenEXR
        import Imath

        header = OpenEXR.Header(w, h)
        pt = Imath.PixelType(Imath.PixelType.HALF)
        header["channels"] = {
            "R": Imath.Channel(pt),
            "G": Imath.Channel(pt),
            "B": Imath.Channel(pt),
            "A": Imath.Channel(pt),
        }

        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels(
            {
                "R": rgba_hw4[:, :, 0].astype(np.float16).tobytes(),
                "G": rgba_hw4[:, :, 1].astype(np.float16).tobytes(),
                "B": rgba_hw4[:, :, 2].astype(np.float16).tobytes(),
                "A": rgba_hw4[:, :, 3].astype(np.float16).tobytes(),
            }
        )
        exr.close()
        return
    except Exception:
        pass


def decoder_input_dim_from_config(cfg) -> int:
    direction_input = getattr(cfg, "decoder_direction_input", "wiwo")
    dir_ch_per_frame = 5 if direction_input == "half_diff" else 6
    return int(cfg.latent_ch + dir_ch_per_frame * cfg.num_frames)


def save_weights_bin(path: Path, weights: dict, output_dim: int = 3, input_dim: int = 20, exp_offset: float = 3.0) -> dict:
    return save_network_weights_bin(
        path=path,
        weights=weights,
        input_dim=input_dim,
        output_dim=output_dim,
        has_frame_linear=True,
        exp_offset=exp_offset,
    )


def _infer_network_layout(
    weights: dict,
    *,
    input_dim: int,
    output_dim: int,
    has_frame_linear: bool = True,
    exp_offset: float | None = None,
) -> dict:
    linear_layers = []
    decoder_layout = {}

    if has_frame_linear:
        frame_weight = np.asarray(weights["frame_linear.weight"], dtype=np.float32)
        if frame_weight.shape != (12, 8):
            raise ValueError(f"frame_linear.weight expected shape (12, 8), got {frame_weight.shape}")
        linear_layers.append("frame_linear")
        decoder_layout["frame_linear.weight"] = [12, 8]

    latent_ch = int(np.asarray(weights.get("latent_ch", np.array([8], dtype=np.int32))).reshape(-1)[0])
    num_frames = int(np.asarray(weights.get("num_frames", np.array([2], dtype=np.int32))).reshape(-1)[0])

    layer_indices = sorted(
        int(k.split(".")[1]) for k in weights.keys() if k.startswith("mlp.") and k.endswith(".weight")
    )
    if not layer_indices:
        raise ValueError("No MLP layers found in weights.")

    linear_layers += [f"mlp.{idx}" for idx in layer_indices]
    mlp_depth = len(layer_indices) - 1
    if mlp_depth not in (2, 3):
        raise ValueError(f"Unsupported MLP depth: {mlp_depth}. Expected 2 or 3.")

    first_w = np.asarray(weights[f"mlp.{layer_indices[0]}.weight"], dtype=np.float32)
    mlp_width = int(first_w.shape[0])

    expected_prev = input_dim
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
    if has_frame_linear and num_frames != 2:
        raise ValueError(f"Only num_frames=2 is supported for runtime export, got {num_frames}")
    if mlp_width not in (16, 32, 64):
        raise ValueError(f"Unsupported mlp_width={mlp_width}. Expected one of 16, 32, 64.")

    return {
        "latent_ch": latent_ch,
        "num_frames": num_frames,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "output_dim": int(output_dim),
        "input_dim": int(input_dim),
        "weight_file_format": "NMDLWT05" if exp_offset is not None else "NMDLWT04",
        "linear_layers": linear_layers,
        "decoder_layout": decoder_layout,
    }


def save_network_weights_bin(
    path: Path,
    weights: dict,
    *,
    input_dim: int,
    output_dim: int,
    has_frame_linear: bool = True,
    exp_offset: float | None = None,
) -> dict:
    layout = _infer_network_layout(
        weights,
        input_dim=input_dim,
        output_dim=output_dim,
        has_frame_linear=has_frame_linear,
        exp_offset=exp_offset,
    )
    latent_ch = layout["latent_ch"]
    num_frames = layout["num_frames"]
    mlp_width = layout["mlp_width"]
    mlp_depth = layout["mlp_depth"]

    with open(path, "wb") as f:
        f.write(b"NMDLWT05" if exp_offset is not None else b"NMDLWT04")
        f.write(struct.pack("<iiiiii", latent_ch, num_frames, mlp_width, mlp_depth, output_dim, input_dim))
        if exp_offset is not None:
            f.write(struct.pack("<f", float(exp_offset)))

        for layer_name in layout["linear_layers"]:
            weight_name = f"{layer_name}.weight"
            bias_name = f"{layer_name}.bias"
            w = np.asarray(weights[weight_name], dtype=np.float32)
            f.write(w.ravel(order="C").tobytes())
            if bias_name in weights:
                b = np.asarray(weights[bias_name], dtype=np.float32)
                f.write(b.ravel(order="C").tobytes())

    return layout


def save_metadata(
    path: Path,
    latent: np.ndarray,
    weights: dict,
    output_dim: int = 3,
    input_dim: int = 20,
    decoder_direction_input: str = "wiwo",
    exp_offset: float = 3.0,
) -> None:
    _, h, w = latent.shape
    layout = _infer_network_layout(
        weights,
        input_dim=input_dim,
        output_dim=output_dim,
        has_frame_linear=True,
        exp_offset=exp_offset,
    )
    metadata = {
        "width": int(w),
        "height": int(h),
        "latent_dim": int(latent.shape[0]),
        "num_frames": layout["num_frames"],
        "apply_exp": True,
        "exp_offset": float(exp_offset),
        "output_dim": layout["output_dim"],
        "input_dim": layout["input_dim"],
        "decoder_direction_input": decoder_direction_input,
        "has_albedo_output": output_dim >= 6,
        "mlp_width": layout["mlp_width"],
        "mlp_depth": layout["mlp_depth"],
        "weight_file_format": layout["weight_file_format"],
        "decoder_layout": layout["decoder_layout"],
        "mip_count": 1,
        "mip_shapes": [[int(h), int(w)]],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_sampler_metadata(path: Path, weights: dict, *, latent_ch: int) -> None:
    # Sampler head outputs {wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy} = 9 channels.
    layout = _infer_network_layout(
        weights,
        input_dim=latent_ch + 3,
        output_dim=9,
        has_frame_linear=False,
    )
    metadata = {
        "latent_dim": layout["latent_ch"],
        "num_frames": layout["num_frames"],
        "mlp_width": layout["mlp_width"],
        "mlp_depth": layout["mlp_depth"],
        "output_dim": 9,
        "weight_file_format": layout["weight_file_format"],
        "decoder_layout": layout["decoder_layout"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def export_renderer_assets(
    model: NeuralMaterialModel,
    cfg: TrainConfig,
    *,
    write_numpy_debug: bool | None = None,
) -> None:
    preview_dir = Path(cfg.preview_out_dir) if cfg.preview_out_dir else Path(__file__).resolve().parents[2] / "MatXScenes" / "Preview"
    os.makedirs(preview_dir, exist_ok=True)
    if write_numpy_debug is None:
        write_numpy_debug = bool(getattr(cfg, "export_numpy_debug", False))

    if cfg.latent_ch != 8:
        print(
            f"[export] Skipping renderer-ready latent/weight bundle because latent_ch={cfg.latent_ch}; expected 8."
        )
        return

    for old_path in list(preview_dir.glob("latent0_mip*.exr")) + list(preview_dir.glob("latent1_mip*.exr")):
        old_path.unlink(missing_ok=True)
    for old_path in list(preview_dir.glob("latent0_mip*.npy")) + list(preview_dir.glob("latent1_mip*.npy")):
        old_path.unlink(missing_ok=True)
    for old_path in (
        preview_dir / "latent0.npy",
        preview_dir / "latent1.npy",
    ):
        old_path.unlink(missing_ok=True)

    latent_levels = [level.detach().cpu().numpy()[0] for level in model.latent.levels]
    for mip, latent in enumerate(latent_levels):
        rgba0 = latent[0:4].transpose(1, 2, 0).copy()
        rgba1 = latent[4:8].transpose(1, 2, 0).copy()

        write_exr(preview_dir / f"latent0_mip{mip}.exr", rgba0)
        write_exr(preview_dir / f"latent1_mip{mip}.exr", rgba1)
        if write_numpy_debug:
            np.save(preview_dir / f"latent0_mip{mip}.npy", rgba0)
            np.save(preview_dir / f"latent1_mip{mip}.npy", rgba1)

        if mip == 0:
            write_exr(preview_dir / "latent0.exr", rgba0)
            write_exr(preview_dir / "latent1.exr", rgba1)
            if write_numpy_debug:
                np.save(preview_dir / "latent0.npy", rgba0)
                np.save(preview_dir / "latent1.npy", rgba1)

    sd = model.decoder.state_dict()
    mlp_layer_indices = sorted(
        int(k.split(".")[1]) for k in sd.keys() if k.startswith("mlp.") and k.endswith(".weight")
    )
    if not mlp_layer_indices:
        raise ValueError("Decoder state_dict contains no mlp.*.weight layers.")
    final_layer = f"mlp.{mlp_layer_indices[-1]}"
    export_albedo = "albedo_head.weight" in sd and "albedo_head.bias" in sd

    weights = {
        "latent_ch": np.array([cfg.latent_ch], dtype=np.int32),
        "num_frames": np.array([cfg.num_frames], dtype=np.int32),
        "frame_linear.weight": sd["frame_linear.weight"].detach().cpu().numpy(),
    }
    for key, value in sd.items():
        if key == f"{final_layer}.weight" and export_albedo:
            brdf_w = value.detach().cpu().numpy()
            albedo_w = sd["albedo_head.weight"].detach().cpu().numpy()
            if brdf_w.shape[1] != albedo_w.shape[1]:
                raise ValueError(
                    f"Cannot export albedo head: final decoder width {brdf_w.shape[1]} "
                    f"does not match albedo head width {albedo_w.shape[1]}."
                )
            weights[key] = np.concatenate([brdf_w, albedo_w], axis=0)
        elif key == f"{final_layer}.bias" and export_albedo:
            brdf_b = value.detach().cpu().numpy()
            albedo_b = sd["albedo_head.bias"].detach().cpu().numpy()
            weights[key] = np.concatenate([brdf_b, albedo_b], axis=0)
        elif key.startswith("mlp."):
            weights[key] = value.detach().cpu().numpy()

    output_dim = 6 if export_albedo else 3
    brdf_input_dim = decoder_input_dim_from_config(cfg)
    save_weights_bin(
        preview_dir / "decoder_weights.bin",
        weights,
        output_dim=output_dim,
        input_dim=brdf_input_dim,
        exp_offset=float(getattr(cfg, "exp_offset", 3.0)),
    )
    save_metadata(
        preview_dir / "metadata.json",
        latent_levels[0],
        weights,
        output_dim=output_dim,
        input_dim=brdf_input_dim,
        decoder_direction_input=getattr(cfg, "decoder_direction_input", "wiwo"),
        exp_offset=float(getattr(cfg, "exp_offset", 3.0)),
    )
    metadata_path = preview_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["mip_count"] = len(latent_levels)
    metadata["mip_shapes"] = [[int(level.shape[1]), int(level.shape[2])] for level in latent_levels]
    metadata["latent_file_pattern"] = "latent{0,1}_mip{mip}.exr"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if cfg.train_importance_sampler:
        sampler_sd = model.importance_sampler.state_dict()
        sampler_weights = {
            "latent_ch": np.array([cfg.latent_ch], dtype=np.int32),
            "num_frames": np.array([cfg.num_frames], dtype=np.int32),
        }
        for key, value in sampler_sd.items():
            if key.startswith("mlp."):
                sampler_weights[key] = value.detach().cpu().numpy()

        save_network_weights_bin(
            preview_dir / "sampler_weights.bin",
            sampler_weights,
            input_dim=cfg.latent_ch + 3,
            output_dim=9,
            has_frame_linear=False,
        )
        save_sampler_metadata(preview_dir / "sampler_metadata.json", sampler_weights, latent_ch=cfg.latent_ch)

    print(f"[export] Renderer-ready assets written to: {preview_dir}")
