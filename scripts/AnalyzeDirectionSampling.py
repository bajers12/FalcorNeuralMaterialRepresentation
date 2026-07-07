#!/usr/bin/env python3
"""
Analyze direction-sampling distributions used for neural-material training.

This script mirrors the OnlineDataGenerationPass direction samplers in Python
and writes report-friendly CSV summaries plus optional PNG plots. It streams in
batches, so tens of millions of samples can be analyzed without storing all
directions in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


EPS = 1e-7
DEG = 180.0 / math.pi
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = REPO_ROOT / "scripts" / "data-generation"


@dataclass
class Histograms:
    theta_i: np.ndarray
    theta_o: np.ndarray
    theta_h: np.ndarray
    theta_d: np.ndarray
    phi_d: np.ndarray
    theta_h_theta_d: np.ndarray
    wi_z: np.ndarray
    wo_z: np.ndarray
    wi_dot_wo: np.ndarray
    wi_polar: np.ndarray
    wo_polar: np.ndarray
    h_polar: np.ndarray
    d_polar: np.ndarray


@dataclass
class BrdfHistograms:
    luminance_log10: np.ndarray
    max_rgb_log10: np.ndarray


@dataclass
class RunningStats:
    count: int = 0
    attempted: int = 0
    invalid: int = 0
    sum_theta_i: float = 0.0
    sum_theta_o: float = 0.0
    sum_theta_h: float = 0.0
    sum_theta_d: float = 0.0
    sum_wi_z: float = 0.0
    sum_wo_z: float = 0.0
    sum_wi_dot_wo: float = 0.0

    def update(
        self,
        *,
        theta_i: np.ndarray,
        theta_o: np.ndarray,
        theta_h: np.ndarray,
        theta_d: np.ndarray,
        wi_z: np.ndarray,
        wo_z: np.ndarray,
        wi_dot_wo: np.ndarray,
        attempted: int,
        invalid: int,
    ) -> None:
        n = int(theta_i.size)
        self.count += n
        self.attempted += int(attempted)
        self.invalid += int(invalid)
        self.sum_theta_i += float(theta_i.sum())
        self.sum_theta_o += float(theta_o.sum())
        self.sum_theta_h += float(theta_h.sum())
        self.sum_theta_d += float(theta_d.sum())
        self.sum_wi_z += float(wi_z.sum())
        self.sum_wo_z += float(wo_z.sum())
        self.sum_wi_dot_wo += float(wi_dot_wo.sum())


@dataclass
class BrdfRunningStats:
    count: int = 0
    zero_luminance_count: int = 0
    sum_luminance: float = 0.0
    sum_max_rgb: float = 0.0
    min_luminance: float = float("inf")
    max_luminance: float = 0.0
    min_max_rgb: float = float("inf")
    max_max_rgb: float = 0.0

    def update(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        luminance = np.maximum(
            0.0,
            0.2126 * y[:, 0].astype(np.float64)
            + 0.7152 * y[:, 1].astype(np.float64)
            + 0.0722 * y[:, 2].astype(np.float64),
        )
        max_rgb = np.maximum(0.0, y.astype(np.float64).max(axis=1))
        self.count += int(y.shape[0])
        self.zero_luminance_count += int(np.count_nonzero(luminance <= 0.0))
        self.sum_luminance += float(luminance.sum())
        self.sum_max_rgb += float(max_rgb.sum())
        if luminance.size:
            self.min_luminance = min(self.min_luminance, float(luminance.min()))
            self.max_luminance = max(self.max_luminance, float(luminance.max()))
            self.min_max_rgb = min(self.min_max_rgb, float(max_rgb.min()))
            self.max_max_rgb = max(self.max_max_rgb, float(max_rgb.max()))
        return luminance, max_rgb


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), EPS)


def make_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sign = np.where(n[:, 2:3] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2:3])
    b = n[:, 0:1] * n[:, 1:2] * a
    tangent = np.concatenate(
        [1.0 + sign * n[:, 0:1] * n[:, 0:1] * a, sign * b, -sign * n[:, 0:1]],
        axis=1,
    )
    bitangent = np.concatenate(
        [b, sign + n[:, 1:2] * n[:, 1:2] * a, -n[:, 1:2]],
        axis=1,
    )
    return normalize(tangent), normalize(bitangent)


def uniform_hemisphere(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.random(n, dtype=np.float64)
    phi = 2.0 * math.pi * rng.random(n, dtype=np.float64)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def cosine_hemisphere(rng: np.random.Generator, n: int) -> np.ndarray:
    u1 = rng.random(n, dtype=np.float64)
    u2 = rng.random(n, dtype=np.float64)
    r = np.sqrt(u1)
    phi = 2.0 * math.pi * u2
    z = np.sqrt(np.maximum(0.0, 1.0 - u1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def theta_phi_hemisphere(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = 0.5 * math.pi * rng.random(n, dtype=np.float64)
    phi = 2.0 * math.pi * rng.random(n, dtype=np.float64)
    sin_theta = np.sin(theta)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=1)


def half_diff_sample(
    rng: np.random.Generator,
    n: int,
    *,
    limited: bool,
    theta_measure: str,
    phi_d_extent: float,
    horizon_eps: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    wi_out: list[np.ndarray] = []
    wo_out: list[np.ndarray] = []
    accepted = 0
    attempted = 0
    invalid = 0

    while accepted < n:
        remaining = n - accepted
        draw_count = max(remaining, min(n, remaining * 2))

        if theta_measure == "theta":
            max_theta_h = math.acos(horizon_eps)
            theta_h = max_theta_h * rng.random(draw_count, dtype=np.float64)
            cos_theta_h = np.cos(theta_h)
        elif theta_measure == "cos_theta":
            cos_theta_h = horizon_eps + (1.0 - horizon_eps) * rng.random(draw_count, dtype=np.float64)
            theta_h = np.arccos(np.clip(cos_theta_h, -1.0, 1.0))
        else:
            raise ValueError(f"Unknown theta measure: {theta_measure}")

        sin_theta_h = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_h * cos_theta_h))
        phi_h = 2.0 * math.pi * rng.random(draw_count, dtype=np.float64)
        h = np.stack(
            [sin_theta_h * np.cos(phi_h), sin_theta_h * np.sin(phi_h), cos_theta_h],
            axis=1,
        )

        h_tangent, h_bitangent = make_basis(h)

        max_theta_d = np.maximum(0.0, math.acos(horizon_eps) - theta_h) if limited else math.acos(horizon_eps)
        if theta_measure == "theta":
            theta_d = max_theta_d * rng.random(draw_count, dtype=np.float64)
            cos_theta_d = np.cos(theta_d)
        else:
            min_cos = np.cos(max_theta_d)
            cos_theta_d = min_cos + (1.0 - min_cos) * rng.random(draw_count, dtype=np.float64)
            theta_d = np.arccos(np.clip(cos_theta_d, -1.0, 1.0))

        sin_theta_d = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_d * cos_theta_d))
        phi_d = phi_d_extent * rng.random(draw_count, dtype=np.float64)

        wi = (
            (sin_theta_d * np.cos(phi_d))[:, None] * h_tangent
            + (sin_theta_d * np.sin(phi_d))[:, None] * h_bitangent
            + cos_theta_d[:, None] * h
        )
        wo = 2.0 * np.sum(wi * h, axis=1, keepdims=True) * h - wi

        attempted += draw_count
        valid = (wi[:, 2] > horizon_eps) & (wo[:, 2] > horizon_eps)
        invalid += int(draw_count - valid.sum())

        wi_valid = normalize(wi[valid])
        wo_valid = normalize(wo[valid])
        take = min(remaining, wi_valid.shape[0])
        if take:
            wi_out.append(wi_valid[:take])
            wo_out.append(wo_valid[:take])
            accepted += take

        if attempted > n * max_attempts and accepted == 0:
            raise RuntimeError("No valid half/difference directions were generated.")

    return np.concatenate(wi_out, axis=0), np.concatenate(wo_out, axis=0), attempted, invalid


def rusinkiewicz_angles(wi: np.ndarray, wo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = normalize(wi + wo)
    theta_h = np.arccos(np.clip(h[:, 2], -1.0, 1.0))
    phi_h = np.mod(np.arctan2(h[:, 1], h[:, 0]), 2.0 * math.pi)
    tangent, bitangent = make_basis(h)
    local_x = np.sum(wi * tangent, axis=1)
    local_y = np.sum(wi * bitangent, axis=1)
    local_z = np.sum(wi * h, axis=1)
    theta_d = np.arccos(np.clip(local_z, -1.0, 1.0))
    phi_d = np.mod(np.arctan2(local_y, local_x), 2.0 * math.pi)
    return theta_h, phi_h, theta_d, phi_d


def make_histograms(theta_bins: int, phi_bins: int) -> Histograms:
    return Histograms(
        theta_i=np.zeros(theta_bins, dtype=np.int64),
        theta_o=np.zeros(theta_bins, dtype=np.int64),
        theta_h=np.zeros(theta_bins, dtype=np.int64),
        theta_d=np.zeros(theta_bins, dtype=np.int64),
        phi_d=np.zeros(phi_bins, dtype=np.int64),
        theta_h_theta_d=np.zeros((theta_bins, theta_bins), dtype=np.int64),
        wi_z=np.zeros(theta_bins, dtype=np.int64),
        wo_z=np.zeros(theta_bins, dtype=np.int64),
        wi_dot_wo=np.zeros(theta_bins, dtype=np.int64),
        wi_polar=np.zeros((theta_bins, phi_bins), dtype=np.int64),
        wo_polar=np.zeros((theta_bins, phi_bins), dtype=np.int64),
        h_polar=np.zeros((theta_bins, phi_bins), dtype=np.int64),
        d_polar=np.zeros((theta_bins, phi_bins), dtype=np.int64),
    )


def make_brdf_histograms(log_bins: int) -> BrdfHistograms:
    return BrdfHistograms(
        luminance_log10=np.zeros(log_bins, dtype=np.int64),
        max_rgb_log10=np.zeros(log_bins, dtype=np.int64),
    )


def bin_1d(values: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    idx = np.floor((np.clip(values, lo, np.nextafter(hi, lo)) - lo) / (hi - lo) * bins).astype(np.int64)
    return np.bincount(idx, minlength=bins)[:bins]


def add_polar_bins(target: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> None:
    radial_bins, angular_bins = target.shape
    half_pi = 0.5 * math.pi
    r_idx = np.floor(
        np.clip(theta, 0.0, np.nextafter(half_pi, 0.0)) / half_pi * radial_bins
    ).astype(np.int64)
    a_idx = np.floor(np.mod(phi, 2.0 * math.pi) / (2.0 * math.pi) * angular_bins).astype(np.int64)
    a_idx = np.clip(a_idx, 0, angular_bins - 1)
    np.add.at(target, (r_idx, a_idx), 1)


def update_histograms(h: Histograms, wi: np.ndarray, wo: np.ndarray) -> tuple[np.ndarray, ...]:
    theta_i = np.arccos(np.clip(wi[:, 2], -1.0, 1.0))
    theta_o = np.arccos(np.clip(wo[:, 2], -1.0, 1.0))
    phi_i = np.mod(np.arctan2(wi[:, 1], wi[:, 0]), 2.0 * math.pi)
    phi_o = np.mod(np.arctan2(wo[:, 1], wo[:, 0]), 2.0 * math.pi)
    theta_h, phi_h, theta_d, phi_d = rusinkiewicz_angles(wi, wo)

    theta_bins = h.theta_i.size
    phi_bins = h.phi_d.size
    half_pi = 0.5 * math.pi

    h.theta_i += bin_1d(theta_i, 0.0, half_pi, theta_bins)
    h.theta_o += bin_1d(theta_o, 0.0, half_pi, theta_bins)
    h.theta_h += bin_1d(theta_h, 0.0, half_pi, theta_bins)
    h.theta_d += bin_1d(theta_d, 0.0, half_pi, theta_bins)
    h.phi_d += bin_1d(phi_d, 0.0, 2.0 * math.pi, phi_bins)
    h.wi_z += bin_1d(wi[:, 2], 0.0, 1.0, theta_bins)
    h.wo_z += bin_1d(wo[:, 2], 0.0, 1.0, theta_bins)
    wi_dot_wo = np.sum(wi * wo, axis=1)
    h.wi_dot_wo += bin_1d(wi_dot_wo, -1.0, 1.0, theta_bins)
    add_polar_bins(h.wi_polar, theta_i, phi_i)
    add_polar_bins(h.wo_polar, theta_o, phi_o)
    add_polar_bins(h.h_polar, theta_h, phi_h)
    add_polar_bins(h.d_polar, theta_d, phi_d)

    th = np.floor(np.clip(theta_h, 0.0, np.nextafter(half_pi, 0.0)) / half_pi * theta_bins).astype(np.int64)
    td = np.floor(np.clip(theta_d, 0.0, np.nextafter(half_pi, 0.0)) / half_pi * theta_bins).astype(np.int64)
    np.add.at(h.theta_h_theta_d, (th, td), 1)
    return theta_i, theta_o, theta_h, theta_d, phi_d, wi_dot_wo


def write_histogram_csv(path: Path, values: np.ndarray, lo: float, hi: float, unit_scale: float = 1.0) -> None:
    total = max(1, int(values.sum()))
    width = (hi - lo) / values.size
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "center", "lo", "hi", "count", "probability"])
        for i, count in enumerate(values):
            bin_lo = lo + i * width
            bin_hi = bin_lo + width
            writer.writerow(
                [
                    i,
                    (bin_lo + 0.5 * width) * unit_scale,
                    bin_lo * unit_scale,
                    bin_hi * unit_scale,
                    int(count),
                    float(count) / total,
                ]
            )


def write_log_histogram_csv(path: Path, values: np.ndarray, log_min: float, log_max: float) -> None:
    write_histogram_csv(path, values, log_min, log_max, 1.0)


def update_log_histogram(target: np.ndarray, values: np.ndarray, log_min: float, log_max: float, eps: float) -> None:
    log_values = np.log10(np.maximum(values, eps))
    target += bin_1d(log_values, log_min, log_max, target.size)


def save_simple_line_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_centers: np.ndarray,
    values: np.ndarray,
) -> None:
    width = 760
    height = 420
    left = 72
    right = 24
    top = 48
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    probabilities = values.astype(np.float64) / max(1, int(values.sum()))
    y_max = max(float(probabilities.max()), EPS)
    x_min = float(x_centers.min()) if x_centers.size else 0.0
    x_max = float(x_centers.max()) if x_centers.size else 1.0
    x_range = max(EPS, x_max - x_min)
    points = []
    for x, y in zip(x_centers, probabilities):
        px = left + plot_w * (float(x) - x_min) / x_range
        py = top + plot_h * (1.0 - float(y) / y_max)
        points.append(f"{px:.2f},{py:.2f}")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 18px sans-serif; fill: #1b1b1b; }}
  .label {{ font: 12px sans-serif; fill: #555; }}
  .axis {{ stroke: #555; stroke-width: 1; }}
  .plot {{ fill: none; stroke: #2454a6; stroke-width: 2.2; }}
</style>
<text x="{width / 2}" y="26" text-anchor="middle" class="title">{title}</text>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
<polyline points="{" ".join(points)}" class="plot"/>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" class="label">{x_label}</text>
<text x="16" y="{height / 2}" text-anchor="middle" class="label" transform="rotate(-90 16 {height / 2})">{y_label}</text>
<text x="{left}" y="{top + plot_h + 18}" text-anchor="middle" class="label">{x_min:.3g}</text>
<text x="{left + plot_w}" y="{top + plot_h + 18}" text-anchor="middle" class="label">{x_max:.3g}</text>
<text x="{left - 8}" y="{top + 4}" text-anchor="end" class="label">{y_max:.3g}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_extra_direction_plots(out_dir: Path, h: Histograms, name: str) -> None:
    dot_centers = np.linspace(-1.0, 1.0, h.wi_dot_wo.size, endpoint=False)
    dot_centers += 1.0 / h.wi_dot_wo.size

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_simple_line_svg(
            out_dir / f"{name}_wi_dot_wo.svg",
            title=f"{name}: dot(wi, wo)",
            x_label="dot(wi, wo)",
            y_label="probability/bin",
            x_centers=dot_centers,
            values=h.wi_dot_wo,
        )
        return

    probability = h.wi_dot_wo.astype(np.float64) / max(1, int(h.wi_dot_wo.sum()))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(dot_centers, probability)
    ax.set_title(f"{name}: dot(wi, wo)")
    ax.set_xlabel("dot(wi, wo)")
    ax.set_ylabel("probability/bin")
    fig.savefig(out_dir / f"{name}_wi_dot_wo.png", dpi=160)
    plt.close(fig)


def save_brdf_plots(out_dir: Path, hist: BrdfHistograms, name: str, args: argparse.Namespace) -> None:
    centers = np.linspace(args.brdf_log_min, args.brdf_log_max, args.brdf_log_bins, endpoint=False)
    centers += 0.5 * (args.brdf_log_max - args.brdf_log_min) / args.brdf_log_bins

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_simple_line_svg(
            out_dir / f"{name}_brdf_luminance_log10.svg",
            title=f"{name}: luminance target distribution",
            x_label="log10(f cos(theta_o))",
            y_label="probability/bin",
            x_centers=centers,
            values=hist.luminance_log10,
        )
        save_simple_line_svg(
            out_dir / f"{name}_brdf_max_rgb_log10.svg",
            title=f"{name}: max RGB target distribution",
            x_label="log10(f cos(theta_o))",
            y_label="probability/bin",
            x_centers=centers,
            values=hist.max_rgb_log10,
        )
        return

    def prob(values: np.ndarray) -> np.ndarray:
        return values.astype(np.float64) / max(1, int(values.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(centers, prob(hist.luminance_log10))
    axes[0].set_title("luminance")
    axes[0].set_xlabel("log10(f cos(theta_o))")
    axes[0].set_ylabel("probability/bin")

    axes[1].plot(centers, prob(hist.max_rgb_log10))
    axes[1].set_title("max RGB")
    axes[1].set_xlabel("log10(f cos(theta_o))")
    axes[1].set_ylabel("probability/bin")

    fig.suptitle(f"{name}: material target distribution")
    fig.savefig(out_dir / f"{name}_brdf_log_histograms.png", dpi=160)
    plt.close(fig)


def write_heatmap_csv(path: Path, values: np.ndarray) -> None:
    total = max(1, int(values.sum()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_h_bin", "theta_d_bin", "count", "probability"])
        for th in range(values.shape[0]):
            for td in range(values.shape[1]):
                count = int(values[th, td])
                writer.writerow([th, td, count, count / total])


def write_polar_csv(path: Path, values: np.ndarray) -> None:
    total = max(1, int(values.sum()))
    radial_bins, angular_bins = values.shape
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "theta_bin",
                "phi_bin",
                "theta_center_deg",
                "phi_center_deg",
                "count",
                "probability",
            ]
        )
        for r in range(radial_bins):
            theta_center = (r + 0.5) * 90.0 / radial_bins
            for a in range(angular_bins):
                phi_center = (a + 0.5) * 360.0 / angular_bins
                count = int(values[r, a])
                writer.writerow([r, a, theta_center, phi_center, count, count / total])


def save_plots(out_dir: Path, h: Histograms, name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_svg_plots(out_dir, h, name)
        print("[warn] matplotlib is not installed; wrote SVG plots instead of PNG plots.")
        return

    theta_centers = (np.arange(h.theta_i.size) + 0.5) * 90.0 / h.theta_i.size
    phi_centers = (np.arange(h.phi_d.size) + 0.5) * 360.0 / h.phi_d.size

    def prob(values: np.ndarray) -> np.ndarray:
        return values / max(1, values.sum())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, title, values in [
        (axes[0, 0], "theta_i", h.theta_i),
        (axes[0, 1], "theta_o", h.theta_o),
        (axes[0, 2], "theta_h", h.theta_h),
        (axes[1, 0], "theta_d", h.theta_d),
    ]:
        ax.plot(theta_centers, prob(values))
        ax.set_title(title)
        ax.set_xlabel("degrees")
        ax.set_ylabel("probability/bin")

    axes[1, 1].plot(phi_centers, prob(h.phi_d))
    axes[1, 1].set_title("phi_d")
    axes[1, 1].set_xlabel("degrees")
    axes[1, 1].set_ylabel("probability/bin")

    image = axes[1, 2].imshow(
        prob(h.theta_h_theta_d).T,
        origin="lower",
        extent=(0, 90, 0, 90),
        aspect="auto",
    )
    axes[1, 2].set_title("theta_h / theta_d")
    axes[1, 2].set_xlabel("theta_h degrees")
    axes[1, 2].set_ylabel("theta_d degrees")
    fig.colorbar(image, ax=axes[1, 2], label="probability")

    fig.suptitle(name)
    fig.savefig(out_dir / f"{name}_histograms.png", dpi=160)
    plt.close(fig)
    save_svg_polar_plots(out_dir, h, name)


def save_svg_plots(out_dir: Path, h: Histograms, name: str) -> None:
    width = 1200
    height = 760
    pad = 54
    panel_w = 360
    panel_h = 210
    gap_x = 28
    gap_y = 70

    def prob(values: np.ndarray) -> np.ndarray:
        total = max(1, int(values.sum()))
        return values.astype(np.float64) / total

    def line_panel(x: int, y: int, title: str, values: np.ndarray, max_x_label: str) -> str:
        p = prob(values)
        pmax = max(float(p.max()), EPS)
        points = []
        for i, value in enumerate(p):
            px = x + pad + (panel_w - 2 * pad) * (i + 0.5) / values.size
            py = y + panel_h - pad - (panel_h - 2 * pad) * float(value) / pmax
            points.append(f"{px:.2f},{py:.2f}")
        axis_y = y + panel_h - pad
        right_x = x + panel_w - pad
        return "\n".join(
            [
                f'<text x="{x + panel_w / 2:.1f}" y="{y + 22}" text-anchor="middle" class="title">{title}</text>',
                f'<line x1="{x + pad}" y1="{axis_y}" x2="{right_x}" y2="{axis_y}" class="axis"/>',
                f'<line x1="{x + pad}" y1="{y + pad}" x2="{x + pad}" y2="{axis_y}" class="axis"/>',
                f'<polyline points="{" ".join(points)}" class="plot"/>',
                f'<text x="{x + pad}" y="{axis_y + 18}" text-anchor="middle" class="tick">0</text>',
                f'<text x="{right_x}" y="{axis_y + 18}" text-anchor="middle" class="tick">{max_x_label}</text>',
                f'<text x="{x + pad - 8}" y="{y + pad + 4}" text-anchor="end" class="tick">{pmax:.3g}</text>',
            ]
        )

    def heatmap_panel(x: int, y: int) -> str:
        p = prob(h.theta_h_theta_d)
        pmax = max(float(p.max()), EPS)
        cell_w = (panel_w - 2 * pad) / p.shape[0]
        cell_h = (panel_h - 2 * pad) / p.shape[1]
        elems = [
            f'<text x="{x + panel_w / 2:.1f}" y="{y + 22}" text-anchor="middle" class="title">theta_h / theta_d</text>'
        ]
        for ix in range(p.shape[0]):
            for iy in range(p.shape[1]):
                value = float(p[ix, iy]) / pmax
                shade = int(255 - 220 * value)
                color = f"rgb({shade},{shade},{255})"
                px = x + pad + ix * cell_w
                py = y + panel_h - pad - (iy + 1) * cell_h
                elems.append(
                    f'<rect x="{px:.2f}" y="{py:.2f}" width="{cell_w + 0.3:.2f}" '
                    f'height="{cell_h + 0.3:.2f}" fill="{color}"/>'
                )
        axis_y = y + panel_h - pad
        right_x = x + panel_w - pad
        elems.extend(
            [
                f'<rect x="{x + pad}" y="{y + pad}" width="{panel_w - 2 * pad}" height="{panel_h - 2 * pad}" class="box"/>',
                f'<text x="{x + pad}" y="{axis_y + 18}" text-anchor="middle" class="tick">0</text>',
                f'<text x="{right_x}" y="{axis_y + 18}" text-anchor="middle" class="tick">90</text>',
                f'<text x="{x + pad - 8}" y="{axis_y}" text-anchor="end" class="tick">0</text>',
                f'<text x="{x + pad - 8}" y="{y + pad + 4}" text-anchor="end" class="tick">90</text>',
            ]
        )
        return "\n".join(elems)

    panels = [
        line_panel(0, 30, "theta_i", h.theta_i, "90"),
        line_panel(panel_w + gap_x, 30, "theta_o", h.theta_o, "90"),
        line_panel(2 * (panel_w + gap_x), 30, "theta_h", h.theta_h, "90"),
        line_panel(0, 30 + panel_h + gap_y, "theta_d", h.theta_d, "90"),
        line_panel(panel_w + gap_x, 30 + panel_h + gap_y, "phi_d", h.phi_d, "360"),
        heatmap_panel(2 * (panel_w + gap_x), 30 + panel_h + gap_y),
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 16px sans-serif; fill: #1b1b1b; }}
  .tick {{ font: 11px sans-serif; fill: #555; }}
  .axis {{ stroke: #555; stroke-width: 1; }}
  .box {{ fill: none; stroke: #555; stroke-width: 1; }}
  .plot {{ fill: none; stroke: #2454a6; stroke-width: 2; }}
</style>
<text x="{width / 2}" y="24" text-anchor="middle" class="title">{name}</text>
{chr(10).join(panels)}
</svg>
"""
    (out_dir / f"{name}_histograms.svg").write_text(svg, encoding="utf-8")
    save_svg_polar_plots(out_dir, h, name)


def save_svg_polar_plots(out_dir: Path, h: Histograms, name: str) -> None:
    size = 980
    panel = 430
    radius = 180
    centers = [(245, 250), (735, 250), (245, 700), (735, 700)]
    fields = [
        ("wi polar", h.wi_polar),
        ("wo polar", h.wo_polar),
        ("h polar", h.h_polar),
        ("d polar", h.d_polar),
    ]

    def color(value: float) -> str:
        value = max(0.0, min(1.0, value))
        r = int(245 - 205 * value)
        g = int(248 - 158 * value)
        b = int(255 - 35 * value)
        return f"rgb({r},{g},{b})"

    def wedge_path(cx: float, cy: float, r0: float, r1: float, a0: float, a1: float) -> str:
        x00 = cx + r0 * math.cos(a0)
        y00 = cy + r0 * math.sin(a0)
        x01 = cx + r0 * math.cos(a1)
        y01 = cy + r0 * math.sin(a1)
        x10 = cx + r1 * math.cos(a0)
        y10 = cy + r1 * math.sin(a0)
        x11 = cx + r1 * math.cos(a1)
        y11 = cy + r1 * math.sin(a1)
        large = 1 if (a1 - a0) > math.pi else 0
        if r0 <= 1e-6:
            return (
                f"M {cx:.2f} {cy:.2f} "
                f"L {x10:.2f} {y10:.2f} "
                f"A {r1:.2f} {r1:.2f} 0 {large} 1 {x11:.2f} {y11:.2f} Z"
            )
        return (
            f"M {x00:.2f} {y00:.2f} "
            f"L {x10:.2f} {y10:.2f} "
            f"A {r1:.2f} {r1:.2f} 0 {large} 1 {x11:.2f} {y11:.2f} "
            f"L {x01:.2f} {y01:.2f} "
            f"A {r0:.2f} {r0:.2f} 0 {large} 0 {x00:.2f} {y00:.2f} Z"
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        "<style>",
        "  .title { font: 18px sans-serif; fill: #1b1b1b; }",
        "  .label { font: 12px sans-serif; fill: #555; }",
        "  .grid { fill: none; stroke: #666; stroke-width: 0.7; opacity: 0.55; }",
        "  .ray { stroke: #666; stroke-width: 0.7; opacity: 0.45; }",
        "</style>",
        f'<text x="{size / 2}" y="28" text-anchor="middle" class="title">{name} polar occupancy</text>',
    ]

    for (title, values), (cx, cy) in zip(fields, centers):
        total = max(1, int(values.sum()))
        probability = values.astype(np.float64) / total
        pmax = max(float(probability.max()), EPS)
        radial_bins, angular_bins = values.shape
        for r in range(radial_bins):
            r0 = radius * r / radial_bins
            r1 = radius * (r + 1) / radial_bins
            for a in range(angular_bins):
                p = float(probability[r, a])
                if p <= 0.0:
                    continue
                a0 = 2.0 * math.pi * a / angular_bins - 0.5 * math.pi
                a1 = 2.0 * math.pi * (a + 1) / angular_bins - 0.5 * math.pi
                elements.append(f'<path d="{wedge_path(cx, cy, r0, r1, a0, a1)}" fill="{color(p / pmax)}"/>')

        for frac, label in [(1 / 3, "30"), (2 / 3, "60"), (1.0, "90")]:
            rr = radius * frac
            elements.append(f'<circle cx="{cx}" cy="{cy}" r="{rr}" class="grid"/>')
            elements.append(f'<text x="{cx + rr + 4}" y="{cy - 3}" class="label">{label}</text>')
        for deg in [0, 90, 180, 270]:
            angle = math.radians(deg) - 0.5 * math.pi
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            elements.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" class="ray"/>')
        elements.append(f'<text x="{cx}" y="{cy - radius - 20}" text-anchor="middle" class="title">{title}</text>')
        elements.append(f'<text x="{cx}" y="{cy + radius + 24}" text-anchor="middle" class="label">radius = theta, center = normal incidence, edge = grazing</text>')

    elements.append("</svg>")
    (out_dir / f"{name}_polar.svg").write_text("\n".join(elements), encoding="utf-8")


def distribution_sampler(name: str, rng: np.random.Generator, n: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, int]:
    if name == "wiwo_uniform":
        return uniform_hemisphere(rng, n), uniform_hemisphere(rng, n), n, 0
    if name == "wiwo_cosine":
        return cosine_hemisphere(rng, n), cosine_hemisphere(rng, n), n, 0
    if name == "wiwo_uniform_theta":
        return theta_phi_hemisphere(rng, n), theta_phi_hemisphere(rng, n), n, 0
    if name == "half_diff":
        return half_diff_sample(
            rng,
            n,
            limited=False,
            theta_measure="theta",
            phi_d_extent=args.phi_d_extent,
            horizon_eps=args.horizon_eps,
            max_attempts=args.max_attempts,
        )
    if name == "half_diff_limited":
        return half_diff_sample(
            rng,
            n,
            limited=True,
            theta_measure="theta",
            phi_d_extent=args.phi_d_extent,
            horizon_eps=args.horizon_eps,
            max_attempts=args.max_attempts,
        )
    if name == "half_diff_cos_theta":
        return half_diff_sample(
            rng,
            n,
            limited=False,
            theta_measure="cos_theta",
            phi_d_extent=args.phi_d_extent,
            horizon_eps=args.horizon_eps,
            max_attempts=args.max_attempts,
        )
    if name == "half_diff_limited_cos_theta":
        return half_diff_sample(
            rng,
            n,
            limited=True,
            theta_measure="cos_theta",
            phi_d_extent=args.phi_d_extent,
            horizon_eps=args.horizon_eps,
            max_attempts=args.max_attempts,
        )
    raise ValueError(f"Unknown distribution: {name}")


def data_generator_config_for_distribution(name: str) -> tuple[str, str] | None:
    """Map analysis distribution names to OnlineDataGenerationPass modes.

    The render pass can evaluate BRDF targets for its native training samplers.
    NumPy-only diagnostic samplers such as wiwo_cosine do not currently have a
    matching material-evaluation mode in the pass.
    """

    if name == "wiwo_uniform":
        return "wiwo", "theta"
    if name == "half_diff":
        return "half_diff", "theta"
    if name == "half_diff_limited":
        return "half_diff_limited", "theta"
    if name == "half_diff_cos_theta":
        return "half_diff", "cos_theta"
    if name == "half_diff_limited_cos_theta":
        return "half_diff_limited", "cos_theta"
    return None


def evaluate_brdf_distribution(name: str, args: argparse.Namespace, out_dir: Path) -> dict | None:
    config = data_generator_config_for_distribution(name)
    if config is None:
        print(
            f"[{name}] skipping BRDF evaluation: no matching OnlineDataGenerationPass sampler mode."
        )
        return None

    if str(DATA_GENERATION_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_GENERATION_DIR))
    from DataGenerator import DataGenerator, SEED_DOMAIN_TRAIN

    direction_sampling, half_diff_theta_measure = config
    brdf_hist = make_brdf_histograms(args.brdf_log_bins)
    brdf_stats = BrdfRunningStats()

    generator = DataGenerator(
        materialId=args.material_id,
        scene_path=str(args.scene_path),
        sampleCount=args.brdf_batch_size,
        bootstrap_feature_layout="none",
        direction_sampling=direction_sampling,
        half_diff_theta_measure=half_diff_theta_measure,
        hierarchical_filtering_enabled=False,
        hierarchical_mip_count=1,
        finest_texture_width=1,
        finest_texture_height=1,
    )

    remaining = args.brdf_samples if args.brdf_samples is not None else args.samples
    generation_index = 0
    while remaining > 0:
        batch_n = min(args.brdf_batch_size, remaining)
        if batch_n != args.brdf_batch_size:
            # The pass sample count is fixed at construction time. Trim the
            # final readback instead of recreating the whole Falcor graph.
            read_n = batch_n
        else:
            read_n = args.brdf_batch_size

        data = np.array(
            generator.generate_data(
                run_seed=args.seed,
                seed_domain=SEED_DOMAIN_TRAIN,
                generation_index=generation_index,
            ),
            copy=True,
        )
        y = data[:read_n, 8:11]
        luminance, max_rgb = brdf_stats.update(y)
        update_log_histogram(
            brdf_hist.luminance_log10,
            luminance,
            args.brdf_log_min,
            args.brdf_log_max,
            args.brdf_log_eps,
        )
        update_log_histogram(
            brdf_hist.max_rgb_log10,
            max_rgb,
            args.brdf_log_min,
            args.brdf_log_max,
            args.brdf_log_eps,
        )

        remaining -= read_n
        generation_index += 1
        if args.progress and (
            brdf_stats.count % args.progress == 0 or remaining == 0
        ):
            print(f"[{name}] brdf={brdf_stats.count:,}/{args.brdf_samples or args.samples:,}")

    write_log_histogram_csv(
        out_dir / "brdf_luminance_log10.csv",
        brdf_hist.luminance_log10,
        args.brdf_log_min,
        args.brdf_log_max,
    )
    write_log_histogram_csv(
        out_dir / "brdf_max_rgb_log10.csv",
        brdf_hist.max_rgb_log10,
        args.brdf_log_min,
        args.brdf_log_max,
    )
    save_brdf_plots(out_dir, brdf_hist, name, args)

    summary = {
        "brdf_distribution": name,
        "brdf_direction_sampling": direction_sampling,
        "brdf_half_diff_theta_measure": half_diff_theta_measure,
        "brdf_samples": brdf_stats.count,
        "brdf_zero_luminance_count": brdf_stats.zero_luminance_count,
        "brdf_zero_luminance_rate": brdf_stats.zero_luminance_count / max(1, brdf_stats.count),
        "brdf_mean_luminance": brdf_stats.sum_luminance / max(1, brdf_stats.count),
        "brdf_mean_max_rgb": brdf_stats.sum_max_rgb / max(1, brdf_stats.count),
        "brdf_min_luminance": 0.0 if brdf_stats.count == 0 else brdf_stats.min_luminance,
        "brdf_max_luminance": brdf_stats.max_luminance,
        "brdf_min_max_rgb": 0.0 if brdf_stats.count == 0 else brdf_stats.min_max_rgb,
        "brdf_max_max_rgb": brdf_stats.max_max_rgb,
        "brdf_log_min": args.brdf_log_min,
        "brdf_log_max": args.brdf_log_max,
        "brdf_log_bins": args.brdf_log_bins,
    }
    (out_dir / "brdf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def analyze_distribution(name: str, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = make_histograms(args.theta_bins, args.phi_bins)
    stats = RunningStats()

    remaining = args.samples
    while remaining > 0:
        batch_n = min(args.batch_size, remaining)
        wi, wo, attempted, invalid = distribution_sampler(name, rng, batch_n, args)
        theta_i, theta_o, theta_h, theta_d, _, wi_dot_wo = update_histograms(hist, wi, wo)
        stats.update(
            theta_i=theta_i,
            theta_o=theta_o,
            theta_h=theta_h,
            theta_d=theta_d,
            wi_z=wi[:, 2],
            wo_z=wo[:, 2],
            wi_dot_wo=wi_dot_wo,
            attempted=attempted,
            invalid=invalid,
        )
        remaining -= batch_n
        if args.progress and (stats.count % args.progress == 0 or remaining == 0):
            print(f"[{name}] accepted={stats.count:,}/{args.samples:,} attempted={stats.attempted:,}")

    half_pi = 0.5 * math.pi
    write_histogram_csv(out_dir / "theta_i.csv", hist.theta_i, 0.0, half_pi, DEG)
    write_histogram_csv(out_dir / "theta_o.csv", hist.theta_o, 0.0, half_pi, DEG)
    write_histogram_csv(out_dir / "theta_h.csv", hist.theta_h, 0.0, half_pi, DEG)
    write_histogram_csv(out_dir / "theta_d.csv", hist.theta_d, 0.0, half_pi, DEG)
    write_histogram_csv(out_dir / "phi_d.csv", hist.phi_d, 0.0, 2.0 * math.pi, DEG)
    write_histogram_csv(out_dir / "wi_z.csv", hist.wi_z, 0.0, 1.0, 1.0)
    write_histogram_csv(out_dir / "wo_z.csv", hist.wo_z, 0.0, 1.0, 1.0)
    write_histogram_csv(out_dir / "wi_dot_wo.csv", hist.wi_dot_wo, -1.0, 1.0, 1.0)
    write_heatmap_csv(out_dir / "theta_h_theta_d.csv", hist.theta_h_theta_d)
    write_polar_csv(out_dir / "wi_polar.csv", hist.wi_polar)
    write_polar_csv(out_dir / "wo_polar.csv", hist.wo_polar)
    write_polar_csv(out_dir / "h_polar.csv", hist.h_polar)
    write_polar_csv(out_dir / "d_polar.csv", hist.d_polar)
    save_plots(out_dir, hist, name)
    save_extra_direction_plots(out_dir, hist, name)

    brdf_summary = evaluate_brdf_distribution(name, args, out_dir) if args.evaluate_brdf else None

    summary = {
        "distribution": name,
        "samples": stats.count,
        "attempted": stats.attempted,
        "invalid_or_rejected": stats.invalid,
        "rejection_rate": stats.invalid / max(1, stats.attempted),
        "acceptance_rate": stats.count / max(1, stats.attempted),
        "mean_theta_i_deg": stats.sum_theta_i / max(1, stats.count) * DEG,
        "mean_theta_o_deg": stats.sum_theta_o / max(1, stats.count) * DEG,
        "mean_theta_h_deg": stats.sum_theta_h / max(1, stats.count) * DEG,
        "mean_theta_d_deg": stats.sum_theta_d / max(1, stats.count) * DEG,
        "mean_wi_z": stats.sum_wi_z / max(1, stats.count),
        "mean_wo_z": stats.sum_wo_z / max(1, stats.count),
        "mean_wi_dot_wo": stats.sum_wi_dot_wo / max(1, stats.count),
        "theta_bins": args.theta_bins,
        "phi_bins": args.phi_bins,
        "phi_d_extent_degrees": args.phi_d_extent * DEG,
        "horizon_eps": args.horizon_eps,
    }
    if brdf_summary is not None:
        summary.update(brdf_summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate histograms for neural-material direction sampling distributions."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=Path("media/LayeredMaterial/ThreeLayeredGGXPreview_NoHeight.pyscene"),
        help="Scene used when --evaluate-brdf is enabled.",
    )
    parser.add_argument("--material-id", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--theta-bins", type=int, default=90)
    parser.add_argument("--phi-bins", type=int, default=180)
    parser.add_argument("--horizon-eps", type=float, default=1e-4)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument(
        "--phi-d-domain",
        choices=("pi", "2pi"),
        default="2pi",
        help="Domain used when drawing phi_d for half/difference samplers.",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=[
            "wiwo_uniform",
            "wiwo_cosine",
            "wiwo_uniform_theta",
            "half_diff",
            "half_diff_limited",
            "half_diff_cos_theta",
            "half_diff_limited_cos_theta",
        ],
    )
    parser.add_argument(
        "--evaluate-brdf",
        action="store_true",
        help=(
            "Also evaluate the actual material target f*cos(theta_o) using "
            "OnlineDataGenerationPass for supported sampling modes."
        ),
    )
    parser.add_argument(
        "--brdf-samples",
        type=int,
        default=None,
        help="Number of material-evaluated samples. Defaults to --samples.",
    )
    parser.add_argument("--brdf-batch-size", type=int, default=262_144)
    parser.add_argument("--brdf-log-bins", type=int, default=160)
    parser.add_argument("--brdf-log-min", type=float, default=-8.0)
    parser.add_argument("--brdf-log-max", type=float, default=2.0)
    parser.add_argument("--brdf-log-eps", type=float, default=1e-8)
    parser.add_argument("--progress", type=int, default=1_000_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.phi_d_extent = math.pi if args.phi_d_domain == "pi" else 2.0 * math.pi
    return args


def main() -> None:
    args = parse_args()
    summaries = [analyze_distribution(name, args) for name in args.distributions]
    summary_path = args.out_dir / "summary.csv"
    keys: list[str] = []
    for summary in summaries:
        for key in summary.keys():
            if key not in keys:
                keys.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[done] wrote sampling analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
