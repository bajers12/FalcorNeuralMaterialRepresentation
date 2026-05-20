import numpy as np
from pathlib import Path
import struct
import json
import os
from OnlineStepfreeze import NeuralMaterialModel, TrainConfig

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
        has_frame_linear=True,
    )


def _infer_network_layout(
    weights: dict,
    *,
    input_dim: int,
    output_dim: int,
    has_frame_linear: bool = True,
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
) -> dict:
    layout = _infer_network_layout(
        weights,
        input_dim=input_dim,
        output_dim=output_dim,
        has_frame_linear=has_frame_linear,
    )
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
    layout = _infer_network_layout(weights, input_dim=20, output_dim=3, has_frame_linear=True)
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

    for old_path in list(preview_dir.glob("latent0_mip*.exr")) + list(preview_dir.glob("latent1_mip*.exr")):
        old_path.unlink(missing_ok=True)
    for old_path in list(preview_dir.glob("latent0_mip*.npy")) + list(preview_dir.glob("latent1_mip*.npy")):
        old_path.unlink(missing_ok=True)

    latent_levels = [level.detach().cpu().numpy()[0] for level in model.latent.levels]
    for mip, latent in enumerate(latent_levels):
        rgba0 = latent[0:4].transpose(1, 2, 0).copy()
        rgba1 = latent[4:8].transpose(1, 2, 0).copy()

        write_exr(preview_dir / f"latent0_mip{mip}.exr", rgba0)
        write_exr(preview_dir / f"latent1_mip{mip}.exr", rgba1)
        np.save(preview_dir / f"latent0_mip{mip}.npy", rgba0)
        np.save(preview_dir / f"latent1_mip{mip}.npy", rgba1)

        if mip == 0:
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
    save_metadata(preview_dir / "metadata.json", latent_levels[0], weights)
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
