#!/usr/bin/env python3
"""
Convert BSDF sampling .bin data into the .npz format expected by LatentMaterialML.py.

Input binary layout (from DataGenerationTest.py):
  uv      : float2
  padding : float2
  wo      : float4
  wi      : float4
  f       : float4

Output .npz keys (for LatentMaterialML.py):
  uv : [N,2] float32
  wi : [N,3] float32
  wo : [N,3] float32
  y  : [N,3] float32   (or rgb if you choose --target-key rgb)
Optional extra keys may also be stored for debugging/traceability.

Notes:
- The trainer only supports .npz and only uses the first 3 channels of wi/wo/y.
- The 4th component of wo/wi/f is ignored by default, but can be kept as debug arrays.
- The trainer docstring says targets are expected to be cos-baked:
      y = f(wi, wo) * max(0, n·wo)
  If your .bin stores raw BSDF values in local space, use --cosine local_wo_z
  so that y = f * max(0, wo.z).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


BSDF_SAMPLE_DTYPE = np.dtype([
    ("uv", "f4", (2,)),
    ("_padding", "f4", (2,)),
    ("wo", "f4", (4,)),
    ("wi", "f4", (4,)),
    ("f", "f4", (4,)),
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert BSDF sample .bin files into LatentMaterialML.py-compatible .npz files."
    )
    p.add_argument("input_bin", type=Path, help="Path to the input .bin file")
    p.add_argument(
        "output_npz",
        type=Path,
        nargs="?",
        help="Path to the output .npz file (default: same name as input, .npz)",
    )
    p.add_argument(
        "--target-key",
        choices=("y", "rgb"),
        default="y",
        help="Name of the target array in the output npz",
    )
    p.add_argument(
        "--cosine",
        choices=("none", "local_wo_z"),
        default="none",
        help=(
            "How to build the target from f. 'none' writes y=f[:, :3]. "
            "'local_wo_z' writes y=f[:, :3] * max(0, wo[:,2])."
        ),
    )
    p.add_argument(
        "--clip-negative-targets",
        action="store_true",
        help="Clamp target values to >= 0 after conversion",
    )
    p.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop samples containing NaN or Inf in uv/wi/wo/target",
    )
    p.add_argument(
        "--save-debug-arrays",
        action="store_true",
        help="Also store wo4, wi4, f4, and dropped 4th channels in the npz",
    )
    p.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Optional path for train split output (.npz)",
    )
    p.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Optional path for validation split output (.npz)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation split ratio in [0,1). Requires --train-output and --val-output if > 0.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Shuffle seed used when writing train/val splits",
    )
    return p.parse_args()


def load_bin(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    data = np.fromfile(path, dtype=BSDF_SAMPLE_DTYPE)
    if data.size == 0:
        raise ValueError(f"No samples loaded from {path}")
    return data


def build_target(f3: np.ndarray, wo3: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        y = f3.copy()
    elif mode == "local_wo_z":
        cos_term = np.maximum(wo3[:, 2:3], 0.0).astype(np.float32)
        y = f3 * cos_term
    else:
        raise ValueError(f"Unsupported cosine mode: {mode}")
    return y.astype(np.float32, copy=False)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        arr2 = arr.reshape(arr.shape[0], -1)
        mask &= np.isfinite(arr2).all(axis=1)
    return mask


def summarize(name: str, arr: np.ndarray) -> str:
    flat = arr.reshape(arr.shape[0], -1)
    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={flat.min():.6g}, max={flat.max():.6g}, mean={flat.mean():.6g}"
    )


def maybe_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    idx = np.arange(n)
    if val_ratio == 0.0:
        return idx, np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    n_val = max(1, n_val) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if train_idx.size == 0:
        raise ValueError("Validation split consumed all samples; lower --val-ratio")
    return train_idx, val_idx


def write_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def subset_payload(payload: dict, idx: np.ndarray) -> dict:
    out = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (payload["uv"].shape[0],):
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()

    input_bin = args.input_bin
    output_npz = args.output_npz or input_bin.with_suffix(".npz")

    if args.val_ratio > 0.0 and (args.train_output is None or args.val_output is None):
        raise ValueError("When using --val-ratio > 0, you must provide both --train-output and --val-output")

    data = load_bin(input_bin)

    uv = data["uv"].astype(np.float32, copy=False)          # [N,2]
    wo4 = data["wo"].astype(np.float32, copy=False)         # [N,4]
    wi4 = data["wi"].astype(np.float32, copy=False)         # [N,4]
    f4 = data["f"].astype(np.float32, copy=False)           # [N,4]

    # Trainer expects 3-vectors for wi/wo/y.
    wo = wo4[:, :3].copy()
    wi = wi4[:, :3].copy()
    f3 = f4[:, :3].copy()

    y = build_target(f3, wo, args.cosine)
    if args.clip_negative_targets:
        y = np.maximum(y, 0.0, dtype=np.float32)

    payload = {
        "uv": uv,
        "wi": wi,
        "wo": wo,
        args.target_key: y,
    }

    if args.save_debug_arrays:
        payload.update({
            "wo4": wo4,
            "wi4": wi4,
            "f4": f4,
            "wo_w": wo4[:, 3].copy(),
            "wi_w": wi4[:, 3].copy(),
            "f_w": f4[:, 3].copy(),
        })

    if args.drop_invalid:
        mask = finite_mask(uv, wi, wo, y)
        dropped = int((~mask).sum())
        if dropped > 0:
            for key, value in list(payload.items()):
                if isinstance(value, np.ndarray) and value.shape[:1] == (mask.shape[0],):
                    payload[key] = value[mask]
            print(f"Dropped {dropped} invalid samples (NaN/Inf)")
        else:
            print("No invalid samples found")

    n = payload["uv"].shape[0]
    if n == 0:
        raise ValueError("No samples remain after filtering")

    print(f"Loaded {len(data)} raw samples from: {input_bin}")
    print(f"Prepared {n} training samples")
    print(f"Target key: {args.target_key}")
    print(f"Cosine mode: {args.cosine}")
    print(summarize("uv", payload["uv"]))
    print(summarize("wi", payload["wi"]))
    print(summarize("wo", payload["wo"]))
    print(summarize(args.target_key, payload[args.target_key]))

    # Single output
    if args.val_ratio == 0.0:
        write_npz(output_npz, payload)
        print(f"Wrote: {output_npz}")
        return

    # Train/val split outputs
    train_idx, val_idx = maybe_split_indices(n, args.val_ratio, args.seed)
    train_payload = subset_payload(payload, train_idx)
    val_payload = subset_payload(payload, val_idx)

    write_npz(args.train_output, train_payload)
    write_npz(args.val_output, val_payload)

    print(f"Wrote train split: {args.train_output} ({train_payload['uv'].shape[0]} samples)")
    print(f"Wrote val split:   {args.val_output} ({val_payload['uv'].shape[0]} samples)")


if __name__ == "__main__":
    main()
