"""Render neural-material runs and compare them against a reference with FLIP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from flip_evaluator import evaluate
except ImportError as exc:
    raise SystemExit(
        "flip-evaluator is required. Install it with: py -3.10 -m pip install flip-evaluator"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOGWAI = REPO_ROOT / "build" / "windows-vs2022" / "bin" / "Release" / "Mogwai.exe"
DEFAULT_RENDER_SCRIPT = REPO_ROOT / "scripts" / "RenderNeuralSphereBatch.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render runtime asset folders with Mogwai, compare ToneMapper captures "
            "against a reference using LDR-FLIP, and merge training validation losses."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Folder containing training directories and their matching *_runtime directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for renders, FLIP maps, CSV metrics, logs, and summary.md.",
    )
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--reference-image",
        type=Path,
        help="Existing tone-mapped PNG reference with the same resolution as the test renders.",
    )
    reference.add_argument(
        "--reference-scene",
        type=Path,
        help="Reference .pyscene to render using the same PathTracer settings.",
    )
    parser.add_argument(
        "--runtime-glob",
        default="*_runtime",
        help="Glob used to discover runtime asset directories under --run-root.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8192,
        help="Accumulated frames per render. Default: 8192.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--reference-name",
        default="reference",
        help="Capture basename when rendering --reference-scene.",
    )
    parser.add_argument(
        "--runtime-suffix",
        default="_runtime",
        help="Suffix removed from runtime directory names to find training folders.",
    )
    parser.add_argument(
        "--mogwai",
        type=Path,
        default=DEFAULT_MOGWAI,
        help=f"Path to Mogwai.exe. Default: {DEFAULT_MOGWAI}",
    )
    parser.add_argument(
        "--render-script",
        type=Path,
        default=DEFAULT_RENDER_SCRIPT,
        help=f"Mogwai batch-render script. Default: {DEFAULT_RENDER_SCRIPT}",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Analyze existing captures in --output-dir without launching Mogwai.",
    )
    parser.add_argument(
        "--render-staging-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "compare_neural_runs_staging",
        help=(
            "Repo-local folder used for Mogwai captures before copying results to "
            "--output-dir. This avoids fragile long renders directly into external folders."
        ),
    )
    parser.add_argument(
        "--launch-delay-seconds",
        type=float,
        default=15.0,
        help="Delay between Mogwai launches. This gives D3D/Falcor teardown time after long renders.",
    )
    parser.add_argument(
        "--sphere-center-x",
        type=float,
        default=0.5,
        help="Sphere-mask center X as a fraction of image width.",
    )
    parser.add_argument(
        "--sphere-center-y",
        type=float,
        default=0.5,
        help="Sphere-mask center Y as a fraction of image height.",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=0.4213,
        help="Sphere-mask radius as a fraction of image height. Default matches the preview scene.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def discover_runtime_dirs(args: argparse.Namespace) -> list[Path]:
    runtime_dirs = sorted(
        path.resolve()
        for path in args.run_root.glob(args.runtime_glob)
        if path.is_dir()
    )
    if not runtime_dirs:
        raise FileNotFoundError(
            f"No runtime directories matching '{args.runtime_glob}' under {args.run_root}"
        )
    return runtime_dirs


def capture_path(output_dir: Path, name: str, frames: int) -> Path:
    return output_dir / f"{name}.ToneMapper.dst.{frames}.png"


def copy_capture(src_dir: Path, dst_dir: Path, name: str, frames: int) -> Path:
    src = require_path(capture_path(src_dir, name, frames), f"Rendered capture for {name}")
    dst = capture_path(dst_dir, name, frames)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_render_logs(src_dir: Path, dst_dir: Path, log_stem: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("mogwai.log", "console.log"):
        src = src_dir / f"{log_stem}.{suffix}"
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)


def configure_mogwai_environment(env: dict[str, str], mogwai_path: Path) -> None:
    bin_dir = mogwai_path.parent
    plugin_dir = bin_dir / "plugins"
    python_dir = bin_dir / "python"

    path_key = "Path" if "Path" in env else "PATH"
    old_path = env.get(path_key, "")
    prepend = [str(bin_dir)]
    if plugin_dir.exists():
        prepend.append(str(plugin_dir))
    env[path_key] = os.pathsep.join(prepend + ([old_path] if old_path else []))

    old_python_path = env.get("PYTHONPATH", "")
    python_paths = [str(python_dir)] if python_dir.exists() else []
    if old_python_path:
        python_paths.append(old_python_path)
    if python_paths:
        env["PYTHONPATH"] = os.pathsep.join(python_paths)


def run_mogwai(
    args: argparse.Namespace,
    env_updates: dict[str, str],
    log_stem: str,
    capture_output_dir: Path,
) -> None:
    capture_output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    configure_mogwai_environment(env, args.mogwai)
    for key in (
        "NEURAL_ASSET_ROOT",
        "NEURAL_ASSET_PATH",
        "NEURAL_ASSET_PATHS",
        "NEURAL_CAPTURE_USE_BSDF_SAMPLING",
        "REFERENCE_SCENE_PATH",
        "REFERENCE_CAPTURE_NAME",
    ):
        env.pop(key, None)
    env.update(env_updates)
    env.update(
        {
            "NEURAL_CAPTURE_OUTPUT": str(capture_output_dir),
            "NEURAL_CAPTURE_FRAMES": str(args.frames),
            "NEURAL_CAPTURE_WIDTH": str(args.width),
            "NEURAL_CAPTURE_HEIGHT": str(args.height),
            "NEURAL_CAPTURE_USE_BSDF_SAMPLING": "0",
        }
    )

    command = [
        str(args.mogwai),
        "--headless",
        "--script",
        str(args.render_script),
        "--logfile",
        str(capture_output_dir / f"{log_stem}.mogwai.log"),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    console_path = capture_output_dir / f"{log_stem}.console.log"
    print(f"[compare] Launching Mogwai: {log_stem}")
    with console_path.open("w", encoding="utf-8") as console:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        copy_render_logs(capture_output_dir, args.output_dir, log_stem)
        raise RuntimeError(
            f"Mogwai failed with exit code {result.returncode}. See {console_path}"
        )
    copy_render_logs(capture_output_dir, args.output_dir, log_stem)


def render_inputs(args: argparse.Namespace, runtime_dirs: list[Path]) -> Path:
    staging_dir = (args.render_staging_dir / args.output_dir.name).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare] Rendering through staging dir: {staging_dir}")

    if args.reference_image:
        reference_path = require_path(args.reference_image, "Reference image")
    else:
        staged_reference = capture_path(staging_dir, args.reference_name, args.frames)
        staged_reference.unlink(missing_ok=True)
        run_mogwai(
            args,
            {
                "REFERENCE_SCENE_PATH": str(require_path(args.reference_scene, "Reference scene")),
                "REFERENCE_CAPTURE_NAME": args.reference_name,
            },
            "reference",
            staging_dir,
        )
        reference_path = copy_capture(staging_dir, args.output_dir, args.reference_name, args.frames)

    if args.launch_delay_seconds > 0.0:
        print(f"[compare] Waiting {args.launch_delay_seconds:g}s before rendering runs.")
        time.sleep(args.launch_delay_seconds)

    for runtime_dir in runtime_dirs:
        run_name = runtime_dir.name.removesuffix(args.runtime_suffix)
        capture_path(staging_dir, run_name, args.frames).unlink(missing_ok=True)
    run_mogwai(
        args,
        {"NEURAL_ASSET_PATHS": os.pathsep.join(str(path) for path in runtime_dirs)},
        "runs",
        staging_dir,
    )
    for runtime_dir in runtime_dirs:
        run_name = runtime_dir.name.removesuffix(args.runtime_suffix)
        copy_capture(staging_dir, args.output_dir, run_name, args.frames)
    return require_path(reference_path, "Rendered reference image")


def find_reference(args: argparse.Namespace) -> Path:
    if args.reference_image:
        return require_path(args.reference_image, "Reference image")
    return require_path(
        capture_path(args.output_dir, args.reference_name, args.frames),
        "Existing rendered reference image",
    )


def load_training_summary(training_dir: Path) -> dict[str, Any]:
    summary_path = training_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_flip_map(error_map: np.ndarray, path: Path) -> None:
    image = np.asarray(error_map)
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0 + 0.5).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    Image.fromarray(image).save(path)


def analyze(
    args: argparse.Namespace,
    runtime_dirs: list[Path],
    reference_path: Path,
) -> list[dict[str, Any]]:
    flip_dir = args.output_dir / "flip_maps"
    flip_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(reference_path) as reference_image:
        width, height = reference_image.size
    yy, xx = np.ogrid[:height, :width]
    center_x = args.sphere_center_x * width
    center_y = args.sphere_center_y * height
    radius = args.sphere_radius * height
    sphere_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2

    rows: list[dict[str, Any]] = []
    for runtime_dir in runtime_dirs:
        run_name = runtime_dir.name.removesuffix(args.runtime_suffix)
        test_path = require_path(
            capture_path(args.output_dir, run_name, args.frames),
            f"Test render for {run_name}",
        )
        print(f"[compare] FLIP: {run_name}")
        error_map_magma, full_mean, _ = evaluate(
            str(reference_path),
            str(test_path),
            "LDR",
            applyMagma=True,
        )
        error_map_raw, _, _ = evaluate(
            str(reference_path),
            str(test_path),
            "LDR",
            applyMagma=False,
            computeMeanError=False,
        )
        raw = np.asarray(error_map_raw)
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        sphere_values = raw[sphere_mask]
        save_flip_map(error_map_magma, flip_dir / f"{run_name}.flip.png")

        training_dir = args.run_root / run_name
        summary = load_training_summary(training_dir)
        row = {
            "run": run_name,
            "status": summary.get("status", ""),
            "best_epoch": summary.get("best_epoch", ""),
            "best_val_loss": summary.get("best_val_loss", ""),
            "final_val_loss": summary.get("last_val_loss", ""),
            "training_hours": (
                float(summary["duration_seconds"]) / 3600.0
                if "duration_seconds" in summary
                else ""
            ),
            "full_flip_mean": float(full_mean),
            "sphere_flip_mean": float(np.mean(sphere_values)),
            "sphere_flip_median": float(np.median(sphere_values)),
            "sphere_flip_p95": float(np.quantile(sphere_values, 0.95)),
            "sphere_flip_p99": float(np.quantile(sphere_values, 0.99)),
            "sphere_flip_max": float(np.max(sphere_values)),
            "sphere_pixels_gt_050": int(np.count_nonzero(sphere_values > 0.5)),
            "sphere_pixels_gt_075": int(np.count_nonzero(sphere_values > 0.75)),
            "render_path": str(test_path),
            "flip_map_path": str(flip_dir / f"{run_name}.flip.png"),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any, digits: int = 6) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    args: argparse.Namespace,
    reference_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    ranked = sorted(rows, key=lambda row: row["sphere_flip_mean"])
    lines = [
        "# Neural Run Comparison",
        "",
        f"- Reference: `{reference_path}`",
        f"- Frames: `{args.frames}`",
        f"- Resolution: `{args.width}x{args.height}`",
        f"- Runtime folders: `{len(rows)}`",
        "",
        "## Ranked By Sphere FLIP Mean",
        "",
        "| Rank | Run | Best val loss | Final val loss | Sphere FLIP mean | P95 | P99 | >0.5 | >0.75 | Training hours |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["run"],
                    format_value(row["best_val_loss"]),
                    format_value(row["final_val_loss"]),
                    format_value(row["sphere_flip_mean"]),
                    format_value(row["sphere_flip_p95"]),
                    format_value(row["sphere_flip_p99"]),
                    str(row["sphere_pixels_gt_050"]),
                    str(row["sphere_pixels_gt_075"]),
                    format_value(row["training_hours"], 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Lower FLIP and validation-loss values are better. Sphere metrics use the configured circular mask.",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.run_root = require_path(args.run_root, "Run root")
    args.mogwai = require_path(args.mogwai, "Mogwai executable")
    args.render_script = require_path(args.render_script, "Mogwai render script")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.render_staging_dir = args.render_staging_dir.resolve()

    if args.frames <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("Frames, width, and height must be positive.")
    if args.sphere_radius <= 0.0:
        raise ValueError("Sphere radius must be positive.")

    runtime_dirs = discover_runtime_dirs(args)
    print(f"[compare] Found {len(runtime_dirs)} runtime directories.")
    reference_path = (
        find_reference(args)
        if args.skip_render
        else render_inputs(args, runtime_dirs)
    )
    rows = analyze(args, runtime_dirs, reference_path)
    rows.sort(key=lambda row: row["sphere_flip_mean"])
    write_csv(rows, args.output_dir / "combined_metrics.csv")
    write_summary(args, reference_path, rows)

    print("")
    print("[compare] Ranking by sphere FLIP mean:")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:2d}. {row['run']}: "
            f"FLIP={row['sphere_flip_mean']:.6f}, "
            f"best_val={format_value(row['best_val_loss'])}"
        )
    print(f"[compare] Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
