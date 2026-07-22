#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Maxwell-Boltzmann analysis for an MD velocity trajectory.

Default system:
    T = 300 K
    N = 100
    L = 100 nm
    Langevin NVT

Input:
    A NumPy velocity trajectory with shape
        (n_frames, n_particles, 3)

Outputs:
    figure1_components_separate.png
        vx, vy, vz in three non-overlapping panels, each compared with
        the theoretical one-dimensional Gaussian distribution.

    figure2_components_overlay.png
        vx, vy, vz overlaid in one panel to compare isotropy.

    figure3_speed_maxwell.png
        Speed histogram compared with the theoretical Maxwell distribution.

    baseline_statistics.csv
        MD mean, theoretical mean, MD standard deviation,
        theoretical standard deviation, errors, and KS statistic.

    baseline_statistics.txt
        Human-readable summary including an anisotropy index.

Examples:
    python analyze_baseline_mb.py

    python analyze_baseline_mb.py \
        --input mb_results_fixed_L100/mb_T300_N100_L100_langevin_vel.npy

    python analyze_baseline_mb.py \
        --input my_simulation_300K_vel.npy \
        --temperature 300 \
        --mass 39.95
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Required on a computing cluster without a display.

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R
from scipy.stats import kstest, maxwell, norm


# ============================================================
# Defaults for the baseline simulation
# ============================================================

DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_MASS_G_MOL = 39.95
DEFAULT_EQUILIBRATION_FRACTION = 0.30
DEFAULT_BINS = 50
DEFAULT_OUTPUT_DIR = Path("baseline_MB_results")

DEFAULT_INPUT_CANDIDATES = (
    Path("mb_results_fixed_L100/mb_T300_N100_L100_langevin_vel.npy"),
    Path("my_simulation_300K_vel.npy"),
)


# ============================================================
# Input handling
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze vx, vy, vz and speed distributions from an MD "
            "velocity trajectory."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Velocity .npy file. If omitted, the program searches the "
            "two default baseline filenames."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE_K,
        help="Target temperature in K. Default: 300.",
    )

    parser.add_argument(
        "--mass",
        type=float,
        default=DEFAULT_MASS_G_MOL,
        help="Molar mass in g/mol. Default: 39.95 for argon.",
    )

    parser.add_argument(
        "--equilibration",
        type=float,
        default=DEFAULT_EQUILIBRATION_FRACTION,
        help="Fraction of initial frames discarded. Default: 0.30.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of histogram bins. Default: 50.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and statistics.",
    )

    return parser.parse_args()


def resolve_input_file(requested_file: Path | None) -> Path:
    if requested_file is not None:
        if not requested_file.is_file():
            raise FileNotFoundError(
                f"Input file does not exist: {requested_file}"
            )
        return requested_file

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.is_file():
            return candidate

    candidate_text = "\n".join(
        f"  - {candidate}" for candidate in DEFAULT_INPUT_CANDIDATES
    )
    raise FileNotFoundError(
        "No default velocity file was found.\n"
        "Expected one of:\n"
        f"{candidate_text}\n"
        "Run the program with --input PATH_TO_FILE.npy."
    )


def load_velocity_data(
    input_file: Path,
    equilibration_fraction: float,
) -> tuple[np.ndarray, int]:
    if not 0.0 <= equilibration_fraction < 1.0:
        raise ValueError("--equilibration must satisfy 0 <= value < 1.")

    velocity = np.load(input_file, allow_pickle=False)

    if velocity.ndim != 3 or velocity.shape[2] != 3:
        raise ValueError(
            f"Unexpected array shape {velocity.shape}. "
            "Expected (n_frames, n_particles, 3)."
        )

    if not np.all(np.isfinite(velocity)):
        raise ValueError("The velocity trajectory contains NaN or infinity.")

    first_analysis_frame = int(
        np.floor(equilibration_fraction * velocity.shape[0])
    )
    stable_velocity = velocity[first_analysis_frame:]

    if stable_velocity.shape[0] < 2:
        raise ValueError(
            "Too few frames remain after discarding equilibration."
        )

    return stable_velocity, first_analysis_frame


# ============================================================
# Theory and statistics
# ============================================================

def component_scale_nm_ps(
    temperature_k: float,
    mass_g_mol: float,
) -> float:
    """
    Theoretical standard deviation of one Cartesian component:

        sigma = sqrt(R T / M)

    R is in J/(mol K), M is converted to kg/mol, and the result
    is converted from m/s to nm/ps.
    """
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive.")
    if mass_g_mol <= 0.0:
        raise ValueError("Molar mass must be positive.")

    mass_kg_mol = mass_g_mol * 1e-3
    sigma_m_s = np.sqrt(R * temperature_k / mass_kg_mol)

    # 1 m/s = 1e-3 nm/ps
    return float(sigma_m_s * 1e-3)


def summarize_distribution(
    name: str,
    values: np.ndarray,
    theoretical_mean: float,
    theoretical_std: float,
    ks_statistic: float,
) -> dict[str, float | int | str]:
    md_mean = float(np.mean(values))
    md_std = float(np.std(values, ddof=1))

    if theoretical_std > 0.0:
        std_relative_error_percent = (
            abs(md_std - theoretical_std)
            / theoretical_std
            * 100.0
        )
    else:
        std_relative_error_percent = float("nan")

    return {
        "distribution": name,
        "n_samples": int(values.size),
        "md_mean_nm_ps": md_mean,
        "theory_mean_nm_ps": theoretical_mean,
        "absolute_mean_error_nm_ps": abs(md_mean - theoretical_mean),
        "md_std_nm_ps": md_std,
        "theory_std_nm_ps": theoretical_std,
        "std_relative_error_percent": std_relative_error_percent,
        "ks_D": float(ks_statistic),
    }


def calculate_statistics(
    components: dict[str, np.ndarray],
    speeds: np.ndarray,
    sigma: float,
) -> tuple[list[dict[str, float | int | str]], float]:
    rows: list[dict[str, float | int | str]] = []

    # vx, vy and vz each follow Normal(0, sigma).
    for name, values in components.items():
        ks_result = kstest(
            values,
            norm.cdf,
            args=(0.0, sigma),
        )

        rows.append(
            summarize_distribution(
                name=name,
                values=values,
                theoretical_mean=0.0,
                theoretical_std=sigma,
                ks_statistic=ks_result.statistic,
            )
        )

    # Speed follows a Maxwell distribution with scale sigma.
    speed_theory_mean = float(
        2.0 * sigma * np.sqrt(2.0 / np.pi)
    )
    speed_theory_std = float(
        sigma * np.sqrt(3.0 - 8.0 / np.pi)
    )

    speed_ks_result = kstest(
        speeds,
        maxwell.cdf,
        args=(0.0, sigma),
    )

    rows.append(
        summarize_distribution(
            name="speed",
            values=speeds,
            theoretical_mean=speed_theory_mean,
            theoretical_std=speed_theory_std,
            ks_statistic=speed_ks_result.statistic,
        )
    )

    component_stds = np.array(
        [np.std(values, ddof=1) for values in components.values()]
    )

    # Dimensionless anisotropy index.
    anisotropy_index = float(
        (np.max(component_stds) - np.min(component_stds))
        / np.mean(component_stds)
    )

    return rows, anisotropy_index


# ============================================================
# Plotting
# ============================================================

def make_common_component_grid(
    components: dict[str, np.ndarray],
    sigma: float,
    n_points: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_component_values = np.concatenate(list(components.values()))

    x_limit = max(
        float(np.max(np.abs(all_component_values))),
        4.0 * sigma,
    )

    x_theory = np.linspace(-x_limit, x_limit, n_points)
    gaussian_pdf = norm.pdf(x_theory, loc=0.0, scale=sigma)

    common_bins = np.linspace(
        -x_limit,
        x_limit,
        DEFAULT_BINS + 1,
    )

    return x_theory, gaussian_pdf, common_bins


def plot_components_separate(
    components: dict[str, np.ndarray],
    sigma: float,
    statistics_by_name: dict[str, dict[str, float | int | str]],
    bins: int,
    output_file: Path,
    title_suffix: str,
) -> None:
    all_component_values = np.concatenate(list(components.values()))
    x_limit = max(
        float(np.max(np.abs(all_component_values))),
        4.0 * sigma,
    )
    x_theory = np.linspace(-x_limit, x_limit, 800)
    theoretical_pdf = norm.pdf(x_theory, loc=0.0, scale=sigma)
    common_bins = np.linspace(-x_limit, x_limit, bins + 1)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
        sharex=True,
        sharey=True,
    )

    for axis, (name, values) in zip(axes, components.items()):
        row = statistics_by_name[name]

        axis.hist(
            values,
            bins=common_bins,
            density=True,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.5,
            label=f"MD {name}",
        )

        axis.plot(
            x_theory,
            theoretical_pdf,
            linestyle="--",
            linewidth=2.2,
            label="Gaussian theory",
        )

        axis.set_title(
            f"${name[0]}_{name[1]}$\n"
            f"$\\mu_{{MD}}={row['md_mean_nm_ps']:.4f}$, "
            f"$\\sigma_{{MD}}={row['md_std_nm_ps']:.4f}$\n"
            f"$D_{{KS}}={row['ks_D']:.4f}$"
        )
        axis.set_xlabel("Velocity component (nm/ps)")
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(bottom=0.0)
        axis.grid(linestyle="--", alpha=0.35)
        axis.legend(fontsize=9)

    axes[0].set_ylabel("Probability density")

    fig.suptitle(
        "Figure 1. Component-wise velocity distributions "
        f"{title_suffix}\n"
        f"Theory: mean = 0, standard deviation = {sigma:.4f} nm/ps",
        fontsize=14,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_components_overlay(
    components: dict[str, np.ndarray],
    sigma: float,
    bins: int,
    anisotropy_index: float,
    output_file: Path,
    title_suffix: str,
) -> None:
    all_component_values = np.concatenate(list(components.values()))
    x_limit = max(
        float(np.max(np.abs(all_component_values))),
        4.0 * sigma,
    )
    x_theory = np.linspace(-x_limit, x_limit, 800)
    theoretical_pdf = norm.pdf(x_theory, loc=0.0, scale=sigma)
    common_bins = np.linspace(-x_limit, x_limit, bins + 1)

    fig, axis = plt.subplots(figsize=(9, 6))

    for name, values in components.items():
        axis.hist(
            values,
            bins=common_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=name,
        )

    axis.plot(
        x_theory,
        theoretical_pdf,
        linestyle="--",
        linewidth=2.4,
        label="Gaussian theory",
    )

    axis.set_title(
        "Figure 2. Overlay of $v_x$, $v_y$, and $v_z$ "
        f"{title_suffix}\n"
        f"Anisotropy index = {anisotropy_index:.4f}"
    )
    axis.set_xlabel("Velocity component (nm/ps)")
    axis.set_ylabel("Probability density")
    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(bottom=0.0)
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_speed_distribution(
    speeds: np.ndarray,
    sigma: float,
    speed_statistics: dict[str, float | int | str],
    bins: int,
    output_file: Path,
    title_suffix: str,
) -> None:
    x_limit = max(
        float(np.max(speeds)),
        5.0 * sigma,
    )
    speed_grid = np.linspace(0.0, x_limit, 1000)
    theoretical_pdf = maxwell.pdf(
        speed_grid,
        loc=0.0,
        scale=sigma,
    )

    fig, axis = plt.subplots(figsize=(9, 6))

    axis.hist(
        speeds,
        bins=bins,
        density=True,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.5,
        label="MD speed",
    )

    axis.plot(
        speed_grid,
        theoretical_pdf,
        linestyle="--",
        linewidth=2.5,
        label="Maxwell theory",
    )

    text = (
        f"MD mean = {speed_statistics['md_mean_nm_ps']:.4f} nm/ps\n"
        f"Theory mean = "
        f"{speed_statistics['theory_mean_nm_ps']:.4f} nm/ps\n"
        f"MD std = {speed_statistics['md_std_nm_ps']:.4f} nm/ps\n"
        f"Theory std = "
        f"{speed_statistics['theory_std_nm_ps']:.4f} nm/ps\n"
        f"KS D = {speed_statistics['ks_D']:.4f}"
    )

    axis.text(
        0.97,
        0.95,
        text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    axis.set_title(
        "Figure 3. Maxwell-Boltzmann speed distribution "
        f"{title_suffix}"
    )
    axis.set_xlabel("Speed (nm/ps)")
    axis.set_ylabel("Probability density")
    axis.set_xlim(0.0, x_limit)
    axis.set_ylim(bottom=0.0)
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Output statistics
# ============================================================

def write_statistics_csv(
    rows: list[dict[str, float | int | str]],
    output_file: Path,
) -> None:
    fieldnames = [
        "distribution",
        "n_samples",
        "md_mean_nm_ps",
        "theory_mean_nm_ps",
        "absolute_mean_error_nm_ps",
        "md_std_nm_ps",
        "theory_std_nm_ps",
        "std_relative_error_percent",
        "ks_D",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_statistics_text(
    rows: list[dict[str, float | int | str]],
    anisotropy_index: float,
    input_file: Path,
    stable_velocity: np.ndarray,
    first_analysis_frame: int,
    temperature_k: float,
    mass_g_mol: float,
    sigma: float,
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("Baseline Maxwell-Boltzmann analysis\n")
        handle.write("=" * 72 + "\n")
        handle.write(f"Input file: {input_file}\n")
        handle.write(f"Temperature: {temperature_k:.6f} K\n")
        handle.write(f"Molar mass: {mass_g_mol:.6f} g/mol\n")
        handle.write(
            f"Particle number read from file: {stable_velocity.shape[1]}\n"
        )
        handle.write(
            f"Frames used: {stable_velocity.shape[0]} "
            f"(starting at frame {first_analysis_frame})\n"
        )
        handle.write(
            f"Theoretical component sigma: {sigma:.8f} nm/ps\n"
        )
        handle.write(
            f"Component anisotropy index: {anisotropy_index:.8f}\n\n"
        )

        header = (
            f"{'name':<10}"
            f"{'MD mean':>14}"
            f"{'theory mean':>16}"
            f"{'MD std':>14}"
            f"{'theory std':>16}"
            f"{'std err %':>13}"
            f"{'KS D':>12}\n"
        )
        handle.write(header)
        handle.write("-" * len(header) + "\n")

        for row in rows:
            handle.write(
                f"{str(row['distribution']):<10}"
                f"{float(row['md_mean_nm_ps']):>14.7f}"
                f"{float(row['theory_mean_nm_ps']):>16.7f}"
                f"{float(row['md_std_nm_ps']):>14.7f}"
                f"{float(row['theory_std_nm_ps']):>16.7f}"
                f"{float(row['std_relative_error_percent']):>13.4f}"
                f"{float(row['ks_D']):>12.6f}\n"
            )

        handle.write("\n")
        handle.write(
            "Interpretation: KS D = sup_x |F_MD(x) - F_theory(x)|. "
            "Smaller values indicate closer agreement.\n"
        )
        handle.write(
            "The trajectory frames are time-correlated; therefore KS D is "
            "used here as a descriptive distance. A formal KS p-value is "
            "not reported.\n"
        )


def print_summary(
    rows: list[dict[str, float | int | str]],
    anisotropy_index: float,
) -> None:
    print("\nStatistics")
    print("-" * 92)
    print(
        f"{'name':<8}"
        f"{'MD mean':>13}"
        f"{'theory mean':>15}"
        f"{'MD std':>13}"
        f"{'theory std':>15}"
        f"{'std error %':>14}"
        f"{'KS D':>11}"
    )

    for row in rows:
        print(
            f"{str(row['distribution']):<8}"
            f"{float(row['md_mean_nm_ps']):>13.6f}"
            f"{float(row['theory_mean_nm_ps']):>15.6f}"
            f"{float(row['md_std_nm_ps']):>13.6f}"
            f"{float(row['theory_std_nm_ps']):>15.6f}"
            f"{float(row['std_relative_error_percent']):>14.3f}"
            f"{float(row['ks_D']):>11.5f}"
        )

    print("-" * 92)
    print(f"Component anisotropy index = {anisotropy_index:.6f}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_arguments()

    try:
        input_file = resolve_input_file(args.input)

        stable_velocity, first_analysis_frame = load_velocity_data(
            input_file=input_file,
            equilibration_fraction=args.equilibration,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        vx = stable_velocity[:, :, 0].reshape(-1)
        vy = stable_velocity[:, :, 1].reshape(-1)
        vz = stable_velocity[:, :, 2].reshape(-1)

        components = {
            "vx": vx,
            "vy": vy,
            "vz": vz,
        }

        speeds = np.linalg.norm(stable_velocity, axis=2).reshape(-1)

        sigma = component_scale_nm_ps(
            temperature_k=args.temperature,
            mass_g_mol=args.mass,
        )

        rows, anisotropy_index = calculate_statistics(
            components=components,
            speeds=speeds,
            sigma=sigma,
        )

        statistics_by_name = {
            str(row["distribution"]): row for row in rows
        }

        n_particles = stable_velocity.shape[1]
        title_suffix = (
            f"($T={args.temperature:g}$ K, "
            f"$N={n_particles}$, Langevin NVT)"
        )

        plot_components_separate(
            components=components,
            sigma=sigma,
            statistics_by_name=statistics_by_name,
            bins=args.bins,
            output_file=(
                args.output_dir
                / "figure1_components_separate.png"
            ),
            title_suffix=title_suffix,
        )

        plot_components_overlay(
            components=components,
            sigma=sigma,
            bins=args.bins,
            anisotropy_index=anisotropy_index,
            output_file=(
                args.output_dir
                / "figure2_components_overlay.png"
            ),
            title_suffix=title_suffix,
        )

        plot_speed_distribution(
            speeds=speeds,
            sigma=sigma,
            speed_statistics=statistics_by_name["speed"],
            bins=args.bins,
            output_file=(
                args.output_dir
                / "figure3_speed_maxwell.png"
            ),
            title_suffix=title_suffix,
        )

        write_statistics_csv(
            rows=rows,
            output_file=args.output_dir / "baseline_statistics.csv",
        )

        write_statistics_text(
            rows=rows,
            anisotropy_index=anisotropy_index,
            input_file=input_file,
            stable_velocity=stable_velocity,
            first_analysis_frame=first_analysis_frame,
            temperature_k=args.temperature,
            mass_g_mol=args.mass,
            sigma=sigma,
            output_file=args.output_dir / "baseline_statistics.txt",
        )

        print(f"Input file: {input_file}")
        print(f"Original array shape: "
              f"({stable_velocity.shape[0] + first_analysis_frame}, "
              f"{stable_velocity.shape[1]}, 3)")
        print(f"First analyzed frame: {first_analysis_frame}")
        print(f"Analyzed array shape: {stable_velocity.shape}")
        print(
            "Theoretical component standard deviation: "
            f"{sigma:.8f} nm/ps"
        )

        if n_particles != 100:
            print(
                f"Warning: this file contains N={n_particles}, "
                "whereas the requested baseline is N=100."
            )

        print_summary(rows, anisotropy_index)

        print(f"\nResults written to: {args.output_dir}")
        print("  figure1_components_separate.png")
        print("  figure2_components_overlay.png")
        print("  figure3_speed_maxwell.png")
        print("  baseline_statistics.csv")
        print("  baseline_statistics.txt")

        return 0

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())