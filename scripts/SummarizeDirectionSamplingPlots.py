#!/usr/bin/env python3
"""
Create report-friendly comparison plots from AnalyzeDirectionSampling.py output.

The polar occupancy plots are useful for debugging azimuthal bias, but for
mostly isotropic sampling distributions the important signal is radial. This
script reads the histogram CSVs and writes compact SVG comparison figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "half_diff": "#1f77b4",
    "half_diff_limited": "#ff7f0e",
    "half_diff_cos_theta": "#9467bd",
    "half_diff_limited_cos_theta": "#8c564b",
    "wiwo_uniform": "#2ca02c",
    "wiwo_cosine": "#d62728",
    "wiwo_uniform_theta": "#17becf",
}


def load_histogram(path: Path) -> tuple[list[float], list[float]]:
    centers: list[float] = []
    probs: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            centers.append(float(row["center"]))
            probs.append(float(row["probability"]))
    return centers, probs


def mass(centers: list[float], probs: list[float], lo: float | None = None, hi: float | None = None) -> float:
    total = 0.0
    for c, p in zip(centers, probs):
        if (lo is None or c >= lo) and (hi is None or c < hi):
            total += p
    return total


def discover_distributions(root: Path) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "theta_h.csv").exists()
    )


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "  .title { font: 18px sans-serif; fill: #1b1b1b; font-weight: 600; }",
        "  .subtitle { font: 13px sans-serif; fill: #555; }",
        "  .axis { stroke: #555; stroke-width: 1; }",
        "  .grid { stroke: #d8d8d8; stroke-width: 1; }",
        "  .tick { font: 11px sans-serif; fill: #555; }",
        "  .legend { font: 12px sans-serif; fill: #222; }",
        "</style>",
    ]


def polyline(points: list[tuple[float, float]], color: str) -> str:
    text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="2.2"/>'


def make_generic_curve_panel(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    x_label: str,
    curves: dict[str, tuple[list[float], list[float]]],
    x_ticks: list[tuple[float, str]] | None = None,
) -> list[str]:
    pad_l = 58
    pad_r = 14
    pad_t = 36
    pad_b = 42
    plot_x = x + pad_l
    plot_y = y + pad_t
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    y_max = max((max(probs) for _, probs in curves.values()), default=1e-9) * 1.08
    y_max = max(y_max, 1e-9)
    x_min = min((min(centers) for centers, _ in curves.values()), default=0.0)
    x_max = max((max(centers) for centers, _ in curves.values()), default=1.0)
    x_span = max(x_max - x_min, 1e-9)

    elems = [
        f'<text x="{x + w / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" class="title">{title}</text>',
    ]
    for frac in [0.25, 0.5, 0.75, 1.0]:
        yy = plot_y + plot_h * (1.0 - frac)
        elems.append(f'<line x1="{plot_x:.1f}" y1="{yy:.1f}" x2="{plot_x + plot_w:.1f}" y2="{yy:.1f}" class="grid"/>')
    elems.extend(
        [
            f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<text x="{plot_x + plot_w / 2:.1f}" y="{plot_y + plot_h + 36:.1f}" text-anchor="middle" class="tick">{x_label}</text>',
            f'<text x="{plot_x - 6:.1f}" y="{plot_y + 4:.1f}" text-anchor="end" class="tick">{y_max:.3f}</text>',
        ]
    )
    if x_ticks is None:
        x_ticks = [(x_min, f"{x_min:.2g}"), (x_max, f"{x_max:.2g}")]
    for value, label in x_ticks:
        if value < x_min - 1e-6 or value > x_max + 1e-6:
            continue
        xx = plot_x + plot_w * (value - x_min) / x_span
        elems.append(f'<line x1="{xx:.1f}" y1="{plot_y + plot_h:.1f}" x2="{xx:.1f}" y2="{plot_y + plot_h + 5:.1f}" class="axis"/>')
        elems.append(f'<text x="{xx:.1f}" y="{plot_y + plot_h + 18:.1f}" text-anchor="middle" class="tick">{label}</text>')
    for name, (centers, probs) in curves.items():
        color = COLORS.get(name, "#333333")
        points = [
            (
                plot_x + plot_w * (c - x_min) / x_span,
                plot_y + plot_h * (1.0 - p / y_max),
            )
            for c, p in zip(centers, probs)
        ]
        elems.append(polyline(points, color))
    return elems


def make_curve_panel(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    curves: dict[str, tuple[list[float], list[float]]],
) -> list[str]:
    pad_l = 52
    pad_r = 12
    pad_t = 34
    pad_b = 34
    plot_x = x + pad_l
    plot_y = y + pad_t
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    y_max = max(max(probs) for _, probs in curves.values()) * 1.08
    y_max = max(y_max, 1e-9)
    elems = [
        f'<text x="{x + w / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" class="title">{title}</text>',
    ]
    for frac in [0.25, 0.5, 0.75, 1.0]:
        yy = plot_y + plot_h * (1.0 - frac)
        elems.append(f'<line x1="{plot_x:.1f}" y1="{yy:.1f}" x2="{plot_x + plot_w:.1f}" y2="{yy:.1f}" class="grid"/>')
    elems.extend(
        [
            f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<text x="{plot_x:.1f}" y="{plot_y + plot_h + 18:.1f}" text-anchor="middle" class="tick">0</text>',
            f'<text x="{plot_x + plot_w:.1f}" y="{plot_y + plot_h + 18:.1f}" text-anchor="middle" class="tick">90 deg</text>',
            f'<text x="{plot_x - 6:.1f}" y="{plot_y + 4:.1f}" text-anchor="end" class="tick">{y_max:.3f}</text>',
        ]
    )
    for name, (centers, probs) in curves.items():
        color = COLORS.get(name, "#333333")
        points = [
            (
                plot_x + plot_w * c / 90.0,
                plot_y + plot_h * (1.0 - p / y_max),
            )
            for c, p in zip(centers, probs)
        ]
        elems.append(polyline(points, color))
    return elems


def make_brdf_dot_figure(root: Path, distributions: list[str], out: Path) -> None:
    brdf_curves = {
        d: load_histogram(root / d / "brdf_luminance_log10.csv")
        for d in distributions
        if (root / d / "brdf_luminance_log10.csv").exists()
    }
    dot_curves = {
        d: load_histogram(root / d / "wi_dot_wo.csv")
        for d in distributions
        if (root / d / "wi_dot_wo.csv").exists()
    }
    if not brdf_curves and not dot_curves:
        return

    width = 1320
    height = 560
    elems = svg_header(width, height)
    elems.append(f'<text x="{width / 2}" y="30" text-anchor="middle" class="title">Material response and pair-angle distributions</text>')
    elems.append(
        f'<text x="{width / 2}" y="52" text-anchor="middle" class="subtitle">'
        "BRDF response uses the material-evaluated training target; dot(wi,wo) shows the angle generated between direction pairs."
        "</text>"
    )

    elems.extend(
        make_generic_curve_panel(
            30,
            82,
            610,
            330,
            "log BRDF target distribution",
            "log10 luminance of f cos(theta_o)",
            brdf_curves,
            x_ticks=[(float(v), str(v)) for v in range(-8, 7, 2)],
        )
    )
    elems.extend(
        make_generic_curve_panel(
            680,
            82,
            610,
            330,
            "dot(wi, wo) distribution",
            "dot(wi, wo)",
            dot_curves,
            x_ticks=[(-1.0, "-1"), (-0.5, "-0.5"), (0.0, "0"), (0.5, "0.5"), (1.0, "1")],
        )
    )

    legend_distributions = [d for d in distributions if d in brdf_curves or d in dot_curves]
    legend_x = 58
    legend_y = height - 90
    for i, d in enumerate(legend_distributions):
        x = legend_x + (i % 4) * 300
        y = legend_y + (i // 4) * 28
        color = COLORS.get(d, "#333333")
        elems.append(f'<line x1="{x}" y1="{y}" x2="{x + 36}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elems.append(f'<text x="{x + 44}" y="{y + 4}" class="legend">{d}</text>')

    elems.append("</svg>")
    out.write_text("\n".join(elems), encoding="utf-8")


def make_curve_figure(root: Path, distributions: list[str], out: Path) -> None:
    fields = [
        ("theta_i", "Incident/view theta"),
        ("theta_o", "Outgoing/light theta"),
        ("theta_h", "Half-vector theta"),
        ("theta_d", "Difference-vector theta"),
    ]
    width = 1320
    height = 920
    elems = svg_header(width, height)
    elems.append(f'<text x="{width / 2}" y="30" text-anchor="middle" class="title">Direction-sampling radial distributions</text>')
    elems.append(
        f'<text x="{width / 2}" y="52" text-anchor="middle" class="subtitle">'
        "These are easier to read than polar maps because the distributions are mostly azimuthally symmetric."
        "</text>"
    )

    for idx, (field, title) in enumerate(fields):
        px = 30 + (idx % 2) * 650
        py = 78 + (idx // 2) * 355
        curves = {
            d: load_histogram(root / d / f"{field}.csv")
            for d in distributions
        }
        elems.extend(make_curve_panel(px, py, 610, 315, title, curves))

    legend_x = 58
    legend_y = height - 98
    for i, d in enumerate(distributions):
        x = legend_x + (i % 4) * 300
        y = legend_y + (i // 4) * 28
        color = COLORS.get(d, "#333333")
        elems.append(f'<line x1="{x}" y1="{y}" x2="{x + 36}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elems.append(f'<text x="{x + 44}" y="{y + 4}" class="legend">{d}</text>')
    elems.append("</svg>")
    out.write_text("\n".join(elems), encoding="utf-8")


def make_mass_figure(root: Path, distributions: list[str], out: Path) -> None:
    metrics = [
        ("theta_h < 10", "theta_h", None, 10.0),
        ("theta_h > 75", "theta_h", 75.0, None),
        ("theta_d < 10", "theta_d", None, 10.0),
        ("theta_d > 75", "theta_d", 75.0, None),
        ("theta_i > 75", "theta_i", 75.0, None),
        ("theta_o > 75", "theta_o", 75.0, None),
    ]
    values: dict[str, list[float]] = {}
    for d in distributions:
        vals = []
        for _, field, lo, hi in metrics:
            centers, probs = load_histogram(root / d / f"{field}.csv")
            vals.append(mass(centers, probs, lo, hi))
        values[d] = vals

    width = 1320
    height = 680
    elems = svg_header(width, height)
    elems.append(f'<text x="{width / 2}" y="30" text-anchor="middle" class="title">Angular mass comparison</text>')
    elems.append(f'<text x="{width / 2}" y="52" text-anchor="middle" class="subtitle">Percent of samples in useful near-specular and grazing regions.</text>')

    chart_x = 90
    chart_y = 95
    chart_w = 1140
    chart_h = 430
    group_w = chart_w / len(metrics)
    bar_w = min(18, group_w / (len(distributions) + 1))
    max_value = max(max(v) for v in values.values()) * 1.12
    max_value = max(max_value, 1e-9)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        y = chart_y + chart_h * (1.0 - frac)
        elems.append(f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" class="grid"/>')
        elems.append(f'<text x="{chart_x - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{100 * max_value * frac:.0f}%</text>')
    elems.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" class="axis"/>')

    for mi, (label, _, _, _) in enumerate(metrics):
        gx = chart_x + mi * group_w
        elems.append(f'<text x="{gx + group_w / 2:.1f}" y="{chart_y + chart_h + 24}" text-anchor="middle" class="tick">{label}</text>')
        for di, d in enumerate(distributions):
            color = COLORS.get(d, "#333333")
            value = values[d][mi]
            x = gx + group_w / 2 - (len(distributions) * bar_w) / 2 + di * bar_w
            h = chart_h * value / max_value
            y = chart_y + chart_h - h
            elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="{color}"/>')

    legend_x = 90
    legend_y = height - 90
    for i, d in enumerate(distributions):
        x = legend_x + (i % 3) * 410
        y = legend_y + (i // 3) * 28
        color = COLORS.get(d, "#333333")
        elems.append(f'<rect x="{x}" y="{y - 10}" width="26" height="12" fill="{color}"/>')
        elems.append(f'<text x="{x + 34}" y="{y}" class="legend">{d}</text>')
    elems.append("</svg>")
    out.write_text("\n".join(elems), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize direction-sampling histograms with readable SVG figures.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--distributions", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "readable_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    distributions = args.distributions or discover_distributions(root)
    make_curve_figure(root, distributions, out_dir / "theta_distribution_comparison.svg")
    make_mass_figure(root, distributions, out_dir / "angular_mass_comparison.svg")
    make_brdf_dot_figure(root, distributions, out_dir / "brdf_dot_distribution_comparison.svg")
    print(f"[done] wrote readable plots to {out_dir}")


if __name__ == "__main__":
    main()
