import argparse
import json
from pathlib import Path

import numpy as np
import torch


def tonemap_for_png(img: np.ndarray, mode: str = "log") -> np.ndarray:
    x = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "clamp":
        x = np.clip(x, 0.0, 1.0)
    elif mode == "log":
        x = np.maximum(x, 0.0)
        x = np.log1p(x)
        m = np.max(x)
        if m > 0:
            x = x / m
    elif mode == "normalize":
        mn = np.min(x)
        mx = np.max(x)
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x)
    else:
        raise ValueError(f"Unknown tonemap mode: {mode}")

    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def write_exr(path: Path, rgb: np.ndarray):
    rgb = np.asarray(rgb, dtype=np.float32)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"Expected HxWx3, got {rgb.shape}"

    h, w, _ = rgb.shape

    try:
        import OpenEXR
        import Imath

        header = OpenEXR.Header(w, h)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        header["channels"] = {
            "R": Imath.Channel(pt),
            "G": Imath.Channel(pt),
            "B": Imath.Channel(pt),
        }

        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels({
            "R": rgb[:, :, 0].astype(np.float32).tobytes(),
            "G": rgb[:, :, 1].astype(np.float32).tobytes(),
            "B": rgb[:, :, 2].astype(np.float32).tobytes(),
        })
        exr.close()
        return
    except Exception as e:
        raise RuntimeError(
            f"Failed to write EXR to {path}. Install OpenEXR.\n{e}"
        )


def write_png(path: Path, rgb_u8: np.ndarray):
    from PIL import Image
    Image.fromarray(rgb_u8, mode="RGB").save(path)


def load_latent_from_pt_or_npz(latent_texture_pt: Path, latent_rgba0_npz: Path, latent_rgba1_npz: Path):
    if latent_rgba0_npz.exists() and latent_rgba1_npz.exists():
        z0 = np.load(latent_rgba0_npz)["rgba"].astype(np.float32)  # 4,H,W
        z1 = np.load(latent_rgba1_npz)["rgba"].astype(np.float32)  # 4,H,W
        latent = np.concatenate([z0, z1], axis=0)                  # 8,H,W
        return latent

    if latent_texture_pt.exists():
        obj = torch.load(latent_texture_pt, map_location="cpu")
        latent = obj["Z"][0].detach().cpu().numpy().astype(np.float32)  # 8,H,W
        return latent

    raise FileNotFoundError("Could not find latent source files.")


def load_state_dict(decoder_pt: Path):
    sd = torch.load(decoder_pt, map_location="cpu")
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy().astype(np.float32)
        else:
            out[k] = np.asarray(v)
    return out


def decode_image_debug(latent_chw: np.ndarray, sd: dict, apply_exp: bool, exp_offset: float):
    # latent_chw: [8,H,W]
    assert latent_chw.shape[0] == 8, f"Expected 8 latent channels, got {latent_chw.shape}"

    z = latent_chw.transpose(1, 2, 0).astype(np.float32)  # H,W,8

    w_frame = sd["frame_linear.weight"].astype(np.float32)  # 12,8
    w0 = sd["mlp.0.weight"].astype(np.float32)              # 32,20
    b0 = sd["mlp.0.bias"].astype(np.float32)                # 32
    w1 = sd["mlp.2.weight"].astype(np.float32)              # 32,32
    b1 = sd["mlp.2.bias"].astype(np.float32)                # 32
    w2 = sd["mlp.4.weight"].astype(np.float32)              # 3,32
    b2 = sd["mlp.4.bias"].astype(np.float32)                # 3

    frame = z @ w_frame.T                     # H,W,12
    x = np.concatenate([z, frame], axis=-1)  # H,W,20

    h0pre = x @ w0.T + b0
    h0 = np.maximum(h0pre, 0.0)

    h1pre = h0 @ w1.T + b1
    h1 = np.maximum(h1pre, 0.0)

    y = h1 @ w2.T + b2  # H,W,3

    if apply_exp:
        decoded = np.exp(y - exp_offset)
    else:
        decoded = y

    tensors = {
        "z": z,
        "frame": frame,
        "x": x,
        "h0pre": h0pre,
        "h0": h0,
        "h1pre": h1pre,
        "h1": h1,
        "y": y,
        "decoded": decoded,
    }
    return tensors

def print_tensor_stats(name: str, arr: np.ndarray):
    flat = arr.reshape(-1, arr.shape[-1])
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    means = flat.mean(axis=0)
    print(f"{name}:")
    print(f"  shape = {arr.shape}")
    print(f"  min   = {mins}")
    print(f"  max   = {maxs}")
    print(f"  mean  = {means}")


def print_tensor_pixel(name: str, arr: np.ndarray, x: int, y: int, max_ch: int = 32):
    h, w, c = arr.shape
    if not (0 <= x < w and 0 <= y < h):
        print(f"{name}[{x},{y}] = out of bounds")
        return

    vals = arr[y, x]
    n = min(c, max_ch)
    print(f"{name}[{x},{y}] first {n}/{c} channels:")
    print(vals[:n])


def save_tensor_npy(out_dir: Path, name: str, arr: np.ndarray):
    np.save(out_dir / f"{name}.npy", arr.astype(np.float32))

def extract_rgb_triplet(arr: np.ndarray, base_ch: int) -> np.ndarray:
    h, w, c = arr.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        ch = base_ch + i
        if ch < c:
            out[:, :, i] = arr[:, :, ch]
    return out

def print_stats(name: str, img: np.ndarray):
    flat = img.reshape(-1, img.shape[-1])
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    means = flat.mean(axis=0)
    print(f"{name}:")
    print(f"  shape = {img.shape}")
    print(f"  min   = {mins}")
    print(f"  max   = {maxs}")
    print(f"  mean  = {means}")


def print_sample_pixels(name: str, img: np.ndarray, coords):
    h, w, _ = img.shape
    print(f"{name} sample pixels:")
    for x, y in coords:
        if 0 <= x < w and 0 <= y < h:
            print(f"  ({x:4d}, {y:4d}) = {img[y, x]}")
        else:
            print(f"  ({x:4d}, {y:4d}) = out of bounds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-pt", required=True)
    ap.add_argument("--latent-texture-pt", default="")
    ap.add_argument("--latent-rgba0-npz", default="")
    ap.add_argument("--latent-rgba1-npz", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp-offset", type=float, default=3.0)
    ap.add_argument("--tonemap", choices=["log", "clamp", "normalize"], default="log")
    ap.add_argument("--debug-x", type=int, default=-1)
    ap.add_argument("--debug-y", type=int, default=-1)
    ap.add_argument("--dump-intermediates", action="store_true")
    ap.add_argument("--debug-tensor", default="y",
                    choices=["z", "frame", "x", "h0pre", "h0", "h1pre", "h1", "y", "decoded"])
    ap.add_argument("--debug-base-channel", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder_pt = Path(args.decoder_pt)
    latent_texture_pt = Path(args.latent_texture_pt) if args.latent_texture_pt else Path("__missing__")
    latent_rgba0_npz = Path(args.latent_rgba0_npz) if args.latent_rgba0_npz else Path("__missing__")
    latent_rgba1_npz = Path(args.latent_rgba1_npz) if args.latent_rgba1_npz else Path("__missing__")

    latent = load_latent_from_pt_or_npz(latent_texture_pt, latent_rgba0_npz, latent_rgba1_npz)
    sd = load_state_dict(decoder_pt)

    tensors_exp = decode_image_debug(latent, sd, apply_exp=True, exp_offset=args.exp_offset)
    tensors_noexp = decode_image_debug(latent, sd, apply_exp=False, exp_offset=args.exp_offset)

    decoded_exp = tensors_exp["decoded"]
    decoded_noexp = tensors_noexp["decoded"]
    raw_y = tensors_noexp["y"]

    print_tensor_stats("decoded_noexp", decoded_noexp)
    print_tensor_stats("decoded_exp", decoded_exp)
    print_tensor_stats("raw_y", raw_y)


    h, w, _ = decoded_exp.shape
    sample_coords = [
        (0, 0),
        (w // 4, h // 4),
        (w // 2, h // 2),
        (3 * w // 4, 3 * h // 4),
        (w - 1, h - 1),
    ]

    print_sample_pixels("decoded_noexp", decoded_noexp, sample_coords)
    print_sample_pixels("decoded_exp", decoded_exp, sample_coords)
    print_sample_pixels("raw_y", raw_y, sample_coords)

    if args.debug_x >= 0 and args.debug_y >= 0:
        print()
        print(f"Detailed tensor dump at pixel ({args.debug_x}, {args.debug_y}):")
        for name in ["z", "frame", "x", "h0pre", "h0", "h1pre", "h1", "y"]:
            print_tensor_pixel(name, tensors_noexp[name], args.debug_x, args.debug_y)
        print_tensor_pixel("decoded", tensors_exp["decoded"], args.debug_x, args.debug_y)

    if args.debug_tensor == "decoded":
        dbg = tensors_exp["decoded"]
    else:
        dbg = tensors_noexp[args.debug_tensor]

    dbg_rgb = extract_rgb_triplet(dbg, args.debug_base_channel)
    np.save(out_dir / "decoded_raw_y.npy", raw_y)
    np.save(out_dir / "decoded_noexp.npy", decoded_noexp)
    np.save(out_dir / "decoded_exp.npy", decoded_exp)

    if args.dump_intermediates:
        for name, arr in tensors_noexp.items():
            save_tensor_npy(out_dir, f"{name}_noexp", arr)

        for name, arr in tensors_exp.items():
            if name == "decoded":
                save_tensor_npy(out_dir, f"{name}_exp", arr)

    write_exr(out_dir / "decoded_noexp.exr", decoded_noexp.astype(np.float32))
    write_exr(out_dir / "decoded_exp.exr", decoded_exp.astype(np.float32))

    png_noexp = tonemap_for_png(decoded_noexp, mode=args.tonemap)
    png_exp = tonemap_for_png(decoded_exp, mode=args.tonemap)

    write_png(out_dir / f"decoded_noexp_{args.tonemap}.png", png_noexp)
    write_png(out_dir / f"decoded_exp_{args.tonemap}.png", png_exp)

    
    dbg = tensors_noexp[args.debug_tensor]
    dbg_rgb = extract_rgb_triplet(dbg, args.debug_base_channel)

    write_exr(out_dir / f"debug_{args.debug_tensor}_ch{args.debug_base_channel:02d}.exr",
              dbg_rgb.astype(np.float32))

    dbg_png = tonemap_for_png(dbg_rgb, mode=args.tonemap)
    write_png(out_dir / f"debug_{args.debug_tensor}_ch{args.debug_base_channel:02d}_{args.tonemap}.png",
              dbg_png)

    metadata = {
        "width": int(w),
        "height": int(h),
        "exp_offset": float(args.exp_offset),
        "tonemap": args.tonemap,
        "sample_coords": [{"x": int(x), "y": int(y)} for x, y in sample_coords],
        "debug_tensor": args.debug_tensor,
        "debug_base_channel": int(args.debug_base_channel),
        "debug_pixel": {
            "x": int(args.debug_x),
            "y": int(args.debug_y),
        },
    }
    with open(out_dir / "reference_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote reference files to: {out_dir}")
    print("  decoded_raw_y.npy")
    print("  decoded_noexp.npy")
    print("  decoded_exp.npy")
    print("  decoded_noexp.exr")
    print("  decoded_exp.exr")
    print(f"  decoded_noexp_{args.tonemap}.png")
    print(f"  decoded_exp_{args.tonemap}.png")
    print("  reference_metadata.json")


if __name__ == "__main__":
    main()