"""Render neural-material runs and compare them against a reference with FLIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
RENDER_TIMINGS_FILENAME = "render_timings.json"


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
        "--neural-scene",
        type=Path,
        default=None,
        help="Optional neural .pyscene for runtime renders. Defaults to MatXScenes/Preview/NeuralSphere_Mosaic.pyscene.",
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
    parser.add_argument(
        "--frame-checkpoints",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help=(
            "Capture and analyze additional accumulated-frame checkpoints in a "
            "single render process. The final --frames value is added automatically."
        ),
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help=(
            "Render this many warmup frames before timing. Use with "
            "--reset-accumulation-between-checkpoints for cleaner low-SPP timing."
        ),
    )
    parser.add_argument(
        "--reset-accumulation-between-checkpoints",
        action="store_true",
        help=(
            "For each frame checkpoint, warm up, reset AccumulatePass, render the "
            "checkpoint from frame 1, and time only that clean render loop."
        ),
    )
    parser.add_argument(
        "--sampler-compare-modes",
        nargs="+",
        choices=("cosine", "learned"),
        default=None,
        help=(
            "Render each runtime folder once per sampler mode. cosine disables "
            "PathTracer BSDF importance sampling, forcing the NeuralMaterial cosine "
            "fallback. learned enables PathTracer BSDF importance sampling, allowing "
            "the NeuralMaterial learned sampler when sampler_weights.bin exists."
        ),
    )
    parser.add_argument(
        "--primary-lod-mode",
        choices=("Mip0", "RayDiffs"),
        default="RayDiffs",
        help=(
            "PathTracer primary-hit texture LOD mode for batch renders. RayDiffs is "
            "required for neural latent mip selection to receive a nonzero footprint."
        ),
    )
    parser.add_argument(
        "--max-surface-bounces",
        type=int,
        default=None,
        help="Optional PathTracer maxSurfaceBounces override for batch renders.",
    )
    parser.add_argument(
        "--latent-mip-debug-mode",
        type=int,
        default=0,
        choices=range(0, 4),
        metavar="{0,1,2,3}",
        help="NeuralMaterial latent mip debug mode. Use 2 to visualize fractional latent LOD as red/green.",
    )
    parser.add_argument(
        "--latent-filtering-mode",
        type=int,
        default=1,
        choices=range(0, 2),
        metavar="{0,1}",
        help="NeuralMaterial latent filtering mode. 1 selects the paper-style stochastic mip path.",
    )
    parser.add_argument(
        "--force-latent-mip",
        action="store_true",
        help="Force NeuralMaterial to use --forced-latent-mip.",
    )
    parser.add_argument(
        "--forced-latent-mip",
        type=int,
        default=0,
        help="Forced latent mip level when --force-latent-mip is set.",
    )
    parser.add_argument(
        "--neural-sampling-mode",
        type=int,
        default=0,
        choices=range(0, 2),
        metavar="{0,1}",
        help="NeuralMaterial sampling mode. 0 is cosine fallback, 1 is learned sampler when available.",
    )
    parser.add_argument(
        "--latent-lod-bias",
        type=float,
        default=0.0,
        help="NeuralMaterial latent LOD bias.",
    )
    parser.add_argument("--camera-pos-x", type=float, default=None)
    parser.add_argument("--camera-pos-y", type=float, default=None)
    parser.add_argument("--camera-pos-z", type=float, default=None)
    parser.add_argument("--camera-target-x", type=float, default=None)
    parser.add_argument("--camera-target-y", type=float, default=None)
    parser.add_argument("--camera-target-z", type=float, default=None)
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
        "--render-runs-individually",
        action="store_true",
        help=(
            "Launch one Mogwai process per runtime asset folder instead of rendering all "
            "runs through NEURAL_ASSET_PATHS in a single process. This is slower, but "
            "avoids loading many 4k latent textures at once."
        ),
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
        path
        for path in args.run_root.glob(args.runtime_glob)
        if path.is_dir()
    )
    if not runtime_dirs:
        raise FileNotFoundError(
            f"No runtime directories matching '{args.runtime_glob}' under {args.run_root}"
        )
    return runtime_dirs


def frame_checkpoints(args: argparse.Namespace) -> list[int]:
    requested = args.frame_checkpoints if args.frame_checkpoints is not None else [args.frames]
    checkpoints = sorted({int(frame) for frame in requested if int(frame) > 0})
    if args.frames not in checkpoints:
        checkpoints.append(args.frames)
    checkpoints = sorted(set(checkpoints))
    if checkpoints[-1] > args.frames:
        raise ValueError("--frame-checkpoints cannot exceed --frames.")
    return checkpoints


def sampler_modes(args: argparse.Namespace) -> list[str]:
    return list(args.sampler_compare_modes) if args.sampler_compare_modes else ["cosine"]


def render_name_for(run_name: str, mode: str, args: argparse.Namespace) -> str:
    if args.sampler_compare_modes is None:
        return run_name
    return f"{run_name}__{mode}"


def timing_key_for(render_name: str, frames: int, checkpoints: list[int]) -> str:
    return render_name if len(checkpoints) == 1 else f"{render_name}@{frames}"


def capture_path(output_dir: Path, name: str, frames: int) -> Path:
    return output_dir / f"{name}.ToneMapper.dst.{frames}.png"


def copy_capture(src_dir: Path, dst_dir: Path, name: str, frames: int) -> Path:
    src = require_path(capture_path(src_dir, name, frames), f"Rendered capture for {name}")
    dst = capture_path(dst_dir, name, frames)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_render_logs(
    src_dir: Path,
    dst_dir: Path,
    source_stem: str,
    destination_stem: str | None = None,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination_stem = destination_stem or source_stem
    for suffix in ("mogwai.log", "console.log"):
        src = src_dir / f"{source_stem}.{suffix}"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{destination_stem}.{suffix}")


def make_mogwai_log_stem(capture_output_dir: Path, requested_stem: str) -> str:
    """Keep Falcor's native logfile path below the legacy Windows path limit."""
    candidate = capture_output_dir / f"{requested_stem}.mogwai.log"
    if len(str(candidate)) < 240:
        return requested_stem
    digest = hashlib.sha1(requested_stem.encode("utf-8")).hexdigest()[:12]
    return f"mogwai_{digest}"


def make_render_staging_dir(root: Path, output_dir: Path) -> Path:
    digest = hashlib.sha1(str(output_dir).encode("utf-8")).hexdigest()[:12]
    return (root / f"comparison_{digest}").resolve()


def copy_render_timings(src_dir: Path, dst_dir: Path) -> None:
    src = src_dir / RENDER_TIMINGS_FILENAME
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def load_render_timings(output_dir: Path) -> dict[str, dict[str, float]]:
    path = output_dir / RENDER_TIMINGS_FILENAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def write_render_timings_csv(output_dir: Path, frames: int | None = None) -> None:
    timings = load_render_timings(output_dir)
    rows = [
        {"render": name, **timing}
        for name, timing in sorted(timings.items())
        if isinstance(timing, dict)
        and (frames is None or int(timing.get("frames", -1)) == frames)
    ]
    if not rows:
        return
    preferred_fieldnames = [
        "render",
        "frames",
        "use_bsdf_sampling",
        "render_seconds",
        "milliseconds_per_frame",
        "frames_per_second",
        "scene_load_seconds",
        "capture_seconds",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in preferred_fieldnames
        }
    )
    fieldnames = preferred_fieldnames + extra_fieldnames
    with (output_dir / "render_timings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "NEURAL_SCENE_PATH",
        "NEURAL_CAPTURE_USE_BSDF_SAMPLING",
        "NEURAL_CAPTURE_FRAME_CHECKPOINTS",
        "NEURAL_CAPTURE_NAME",
        "NEURAL_CAPTURE_MODE_SUFFIX",
        "NEURAL_PRIMARY_LOD_MODE",
        "NEURAL_MAX_SURFACE_BOUNCES",
        "NEURAL_LATENT_MIP_DEBUG_MODE",
        "NEURAL_LATENT_FILTERING_MODE",
        "NEURAL_FORCE_LATENT_MIP",
        "NEURAL_FORCED_LATENT_MIP",
        "NEURAL_SAMPLING_MODE",
        "NEURAL_LATENT_LOD_BIAS",
        "NEURAL_CAMERA_POS_X",
        "NEURAL_CAMERA_POS_Y",
        "NEURAL_CAMERA_POS_Z",
        "NEURAL_CAMERA_TARGET_X",
        "NEURAL_CAMERA_TARGET_Y",
        "NEURAL_CAMERA_TARGET_Z",
        "REFERENCE_SCENE_PATH",
        "REFERENCE_CAPTURE_NAME",
    ):
        env.pop(key, None)
    env.update(
        {
            "NEURAL_CAPTURE_OUTPUT": str(capture_output_dir),
            "NEURAL_CAPTURE_FRAMES": str(args.frames),
            "NEURAL_CAPTURE_FRAME_CHECKPOINTS": " ".join(
                str(frame) for frame in frame_checkpoints(args)
            ),
            "NEURAL_CAPTURE_WIDTH": str(args.width),
            "NEURAL_CAPTURE_HEIGHT": str(args.height),
            "NEURAL_CAPTURE_USE_BSDF_SAMPLING": "0",
            "NEURAL_CAPTURE_WARMUP_FRAMES": str(max(0, args.warmup_frames)),
            "NEURAL_CAPTURE_RESET_BETWEEN_CHECKPOINTS": "1"
            if args.reset_accumulation_between_checkpoints
            else "0",
            "NEURAL_PRIMARY_LOD_MODE": args.primary_lod_mode,
            "NEURAL_LATENT_MIP_DEBUG_MODE": str(args.latent_mip_debug_mode),
            "NEURAL_LATENT_FILTERING_MODE": str(args.latent_filtering_mode),
            "NEURAL_FORCE_LATENT_MIP": "1" if args.force_latent_mip else "0",
            "NEURAL_FORCED_LATENT_MIP": str(max(0, args.forced_latent_mip)),
            "NEURAL_SAMPLING_MODE": str(args.neural_sampling_mode),
            "NEURAL_LATENT_LOD_BIAS": str(args.latent_lod_bias),
        }
    )
    optional_env = {
        "NEURAL_CAMERA_POS_X": args.camera_pos_x,
        "NEURAL_CAMERA_POS_Y": args.camera_pos_y,
        "NEURAL_CAMERA_POS_Z": args.camera_pos_z,
        "NEURAL_CAMERA_TARGET_X": args.camera_target_x,
        "NEURAL_CAMERA_TARGET_Y": args.camera_target_y,
        "NEURAL_CAMERA_TARGET_Z": args.camera_target_z,
        "NEURAL_MAX_SURFACE_BOUNCES": args.max_surface_bounces,
        "NEURAL_SCENE_PATH": require_path(args.neural_scene, "Neural scene") if args.neural_scene else None,
    }
    env.update({key: str(value) for key, value in optional_env.items() if value is not None})
    env.update(env_updates)
    file_log_stem = make_mogwai_log_stem(capture_output_dir, log_stem)

    command = [
        str(args.mogwai),
        "--headless",
        "--script",
        str(args.render_script),
        "--logfile",
        str(capture_output_dir / f"{file_log_stem}.mogwai.log"),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    console_path = capture_output_dir / f"{file_log_stem}.console.log"
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
    copy_render_timings(capture_output_dir, args.output_dir)
    if result.returncode != 0:
        copy_render_logs(capture_output_dir, args.output_dir, file_log_stem, log_stem)
        raise RuntimeError(
            f"Mogwai failed with exit code {result.returncode}. See {console_path}"
        )
    copy_render_logs(capture_output_dir, args.output_dir, file_log_stem, log_stem)


def render_inputs(args: argparse.Namespace, runtime_dirs: list[Path]) -> Path:
    staging_dir = make_render_staging_dir(args.render_staging_dir, args.output_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / RENDER_TIMINGS_FILENAME).unlink(missing_ok=True)
    print(f"[compare] Rendering through staging dir: {staging_dir}")
    checkpoints = frame_checkpoints(args)
    modes = sampler_modes(args)

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
        for mode in modes:
            render_name = render_name_for(run_name, mode, args)
            for frames in checkpoints:
                capture_path(staging_dir, render_name, frames).unlink(missing_ok=True)

    if args.render_runs_individually:
        for runtime_dir in runtime_dirs:
            run_name = runtime_dir.name.removesuffix(args.runtime_suffix)
            for mode in modes:
                render_name = render_name_for(run_name, mode, args)
                run_mogwai(
                    args,
                    {
                        "NEURAL_ASSET_PATH": str(runtime_dir),
                        "NEURAL_CAPTURE_USE_BSDF_SAMPLING": "1" if mode == "learned" else "0",
                        "NEURAL_CAPTURE_NAME": render_name,
                    },
                    f"run_{render_name}",
                    staging_dir,
                )
                if args.launch_delay_seconds > 0.0:
                    print(f"[compare] Waiting {args.launch_delay_seconds:g}s before next run.")
                    time.sleep(args.launch_delay_seconds)
    else:
        if len(modes) == 1 and args.sampler_compare_modes is None:
            run_mogwai(
                args,
                {
                    "NEURAL_ASSET_PATHS": os.pathsep.join(str(path) for path in runtime_dirs),
                    "NEURAL_CAPTURE_USE_BSDF_SAMPLING": "1" if modes[0] == "learned" else "0",
                },
                "runs",
                staging_dir,
            )
        else:
            for mode in modes:
                run_mogwai(
                    args,
                    {
                        "NEURAL_ASSET_PATHS": os.pathsep.join(str(path) for path in runtime_dirs),
                        "NEURAL_CAPTURE_USE_BSDF_SAMPLING": "1" if mode == "learned" else "0",
                        "NEURAL_CAPTURE_MODE_SUFFIX": mode,
                    },
                    f"runs_{mode}",
                    staging_dir,
                )
                if args.launch_delay_seconds > 0.0:
                    print(f"[compare] Waiting {args.launch_delay_seconds:g}s before next sampler mode.")
                    time.sleep(args.launch_delay_seconds)
    for runtime_dir in runtime_dirs:
        run_name = runtime_dir.name.removesuffix(args.runtime_suffix)
        for mode in modes:
            render_name = render_name_for(run_name, mode, args)
            for frames in checkpoints:
                copy_capture(staging_dir, args.output_dir, render_name, frames)
    copy_render_timings(staging_dir, args.output_dir)
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
    render_timings = load_render_timings(args.output_dir)
    checkpoints = frame_checkpoints(args)
    modes = sampler_modes(args)

    rows: list[dict[str, Any]] = []
    for runtime_dir in runtime_dirs:
        run_name = runtime_dir.name.removesuffix(args.runtime_suffix)
        training_dir = args.run_root / run_name
        summary = load_training_summary(training_dir)
        for mode in modes:
            render_name = render_name_for(run_name, mode, args)
            for frames in checkpoints:
                test_path = require_path(
                    capture_path(args.output_dir, render_name, frames),
                    f"Test render for {render_name} at {frames} frames",
                )
                print(f"[compare] FLIP: {render_name} @ {frames}")
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
                flip_map_path = flip_dir / f"{render_name}.{frames}.flip.png"
                save_flip_map(error_map_magma, flip_map_path)

                render_timing = render_timings.get(
                    timing_key_for(render_name, frames, checkpoints), {}
                )
                if int(render_timing.get("frames", -1)) != frames:
                    render_timing = {}
                row = {
                    "run": render_name,
                    "base_run": run_name,
                    "sampler_mode": mode,
                    "frames": frames,
                    "status": summary.get("status", ""),
                    "best_epoch": summary.get("best_epoch", ""),
                    "best_val_loss": summary.get("best_val_loss", ""),
                    "final_val_loss": summary.get("last_val_loss", ""),
                    "sampler_loss": summary.get("sampler_loss", ""),
                    "training_hours": (
                        float(summary["duration_seconds"]) / 3600.0
                        if "duration_seconds" in summary
                        else ""
                    ),
                    "render_seconds": render_timing.get("render_seconds", ""),
                    "render_ms_per_frame": render_timing.get("milliseconds_per_frame", ""),
                    "render_fps": render_timing.get("frames_per_second", ""),
                    "scene_load_seconds": render_timing.get("scene_load_seconds", ""),
                    "capture_seconds": render_timing.get("capture_seconds", ""),
                    "full_flip_mean": float(full_mean),
                    "sphere_flip_mean": float(np.mean(sphere_values)),
                    "sphere_flip_median": float(np.median(sphere_values)),
                    "sphere_flip_p95": float(np.quantile(sphere_values, 0.95)),
                    "sphere_flip_p99": float(np.quantile(sphere_values, 0.99)),
                    "sphere_flip_max": float(np.max(sphere_values)),
                    "sphere_pixels_gt_050": int(np.count_nonzero(sphere_values > 0.5)),
                    "sphere_pixels_gt_075": int(np.count_nonzero(sphere_values > 0.75)),
                    "render_path": str(test_path),
                    "flip_map_path": str(flip_map_path),
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
    render_timings = load_render_timings(args.output_dir)
    reference_timing = render_timings.get(args.reference_name, {})
    lines = [
        "# Neural Run Comparison",
        "",
        f"- Reference: `{reference_path}`",
        f"- Frames: `{args.frames}`",
        f"- Frame checkpoints: `{frame_checkpoints(args)}`",
        f"- Sampler modes: `{sampler_modes(args)}`",
        f"- Resolution: `{args.width}x{args.height}`",
        f"- Runtime folders: `{len(rows)}`",
    ]
    if int(reference_timing.get("frames", -1)) == args.frames:
        lines.append(
            f"- Reference render loop: `{float(reference_timing['render_seconds']):.3f}s` "
            f"(`{float(reference_timing['milliseconds_per_frame']):.3f}ms/frame`)"
        )
    lines.extend(
        [
            "",
            "## Ranked By Sphere FLIP Mean",
            "",
            "| Rank | Run | Mode | Frames | Best val loss | Final val loss | Sampler loss | Sphere FLIP mean | P95 | P99 | >0.5 | >0.75 | Render seconds | ms/frame | FPS | Training hours |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["run"],
                    row.get("sampler_mode", ""),
                    str(row.get("frames", "")),
                    format_value(row["best_val_loss"]),
                    format_value(row["final_val_loss"]),
                    format_value(row.get("sampler_loss", "")),
                    format_value(row["sphere_flip_mean"]),
                    format_value(row["sphere_flip_p95"]),
                    format_value(row["sphere_flip_p99"]),
                    str(row["sphere_pixels_gt_050"]),
                    str(row["sphere_pixels_gt_075"]),
                    format_value(row["render_seconds"], 3),
                    format_value(row["render_ms_per_frame"], 3),
                    format_value(row["render_fps"], 3),
                    format_value(row["training_hours"], 3),
                ]
            )
            + " |"
        )

    if args.sampler_compare_modes is not None or len(frame_checkpoints(args)) > 1:
        checkpoint_ranked = sorted(
            rows,
            key=lambda row: (
                str(row.get("base_run", row.get("run", ""))),
                int(row.get("frames", 0)),
                str(row.get("sampler_mode", "")),
            ),
        )
        lines.extend(
            [
                "",
                "## Sampler Time-To-Quality",
                "",
                "| Run | Mode | Frames | Sphere FLIP mean | P95 | P99 | Render seconds | ms/frame | FPS |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in checkpoint_ranked:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["base_run"],
                        row.get("sampler_mode", ""),
                        str(row.get("frames", "")),
                        format_value(row["sphere_flip_mean"]),
                        format_value(row["sphere_flip_p95"]),
                        format_value(row["sphere_flip_p99"]),
                        format_value(row["render_seconds"], 3),
                        format_value(row["render_ms_per_frame"], 3),
                        format_value(row["render_fps"], 3),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "Lower FLIP and validation-loss values are better. Sphere metrics use the configured circular mask.",
            "Render timing is high-resolution wall time around the Mogwai `renderFrame()` loop; it excludes scene loading, capture writing, launch delays, and FLIP analysis.",
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
    write_render_timings_csv(args.output_dir, None)
    write_summary(args, reference_path, rows)

    print("")
    print("[compare] Ranking by sphere FLIP mean:")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:2d}. {row['run']}: "
            f"mode={row.get('sampler_mode', '')}, "
            f"frames={row.get('frames', '')}, "
            f"FLIP={row['sphere_flip_mean']:.6f}, "
            f"best_val={format_value(row['best_val_loss'])}, "
            f"render={format_value(row['render_seconds'], 3)}s"
        )
    print(f"[compare] Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
