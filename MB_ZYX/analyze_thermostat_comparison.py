#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Langevin, Andersen, Berendsen and NVE velocity trajectories.

Fixed comparison:
    T_target = 300 K
    N = 200
    L = 100 nm

Required files:
    mb_results/mb_T300_N200_langevin_vel.npy
    mb_results/mb_T300_N200_andersen_vel.npy
    mb_results/mb_T300_N200_berendsen_vel.npy
    mb_results/mb_T300_N200_nve_vel.npy
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R
from scipy.stats import maxwell


TARGET_TEMPERATURE_K = 300.0
MASS_ARGON_G_MOL = 39.95
N_PARTICLES = 200
DT_PS = 0.1
EQUILIBRATION_FRACTION = 0.30
N_BINS = 60

METHODS = ("langevin", "andersen", "berendsen", "nve")
INPUT_FILES = {
    method: Path(f"mb_results/mb_T300_N200_{method}_vel.npy")
    for method in METHODS
}
OUTPUT_DIR = Path("thermostat_comparison_results")


def theoretical_scale_nm_ps(temperature_k: float) -> float:
    mass_kg_mol = MASS_ARGON_G_MOL * 1e-3
    return float(np.sqrt(R * temperature_k / mass_kg_mol) * 1e-3)


def instantaneous_temperature(velocity: np.ndarray) -> np.ndarray:
    """Return T(t), using the same 3N convention as LJ_gas.py."""
    sum_v2 = np.sum(velocity**2, axis=(1, 2))
    kinetic_energy_kj_mol = 0.5 * MASS_ARGON_G_MOL * sum_v2
    return (
        2.0 * kinetic_energy_kj_mol * 1e3
        / (3.0 * velocity.shape[1] * R)
    )


def ks_distance(sample: np.ndarray, scale: float) -> float:
    values = np.sort(sample)
    n_values = values.size
    upper = np.arange(1, n_values + 1, dtype=float) / n_values
    lower = np.arange(0, n_values, dtype=float) / n_values
    theory = maxwell.cdf(values, loc=0.0, scale=scale)
    return float(max(
        np.max(np.abs(upper - theory)),
        np.max(np.abs(lower - theory)),
    ))


def framewise_ks(speeds_by_frame: np.ndarray, scale: float) -> np.ndarray:
    return np.array([
        ks_distance(frame, scale)
        for frame in speeds_by_frame
    ])


def load_method(method: str) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    input_file = INPUT_FILES[method]
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Missing {input_file}. Run the {method} trajectory first."
        )

    velocity = np.load(input_file, allow_pickle=False)
    if velocity.ndim != 3 or velocity.shape[2] != 3:
        raise ValueError(
            f"{input_file} has shape {velocity.shape}; expected (frames, N, 3)."
        )
    if velocity.shape[1] != N_PARTICLES:
        raise ValueError(
            f"{input_file} contains N={velocity.shape[1]}, expected {N_PARTICLES}."
        )
    if not np.all(np.isfinite(velocity)):
        raise ValueError(f"{input_file} contains NaN or infinity.")

    first_frame = int(EQUILIBRATION_FRACTION * velocity.shape[0])
    stable_velocity = velocity[first_frame:]
    speeds_by_frame = np.linalg.norm(stable_velocity, axis=2)
    pooled_speeds = speeds_by_frame.reshape(-1)
    temperatures = instantaneous_temperature(stable_velocity)

    mean_temperature = float(np.mean(temperatures))
    temperature_std = float(np.std(temperatures, ddof=1))
    relative_variance = float(
        np.var(temperatures, ddof=1) / mean_temperature**2
    )
    canonical_relative_variance = 2.0 / (3.0 * N_PARTICLES)
    fluctuation_ratio = relative_variance / canonical_relative_variance

    target_scale = theoretical_scale_nm_ps(TARGET_TEMPERATURE_K)
    measured_scale = theoretical_scale_nm_ps(mean_temperature)

    frame_ks_target = framewise_ks(speeds_by_frame, target_scale)
    pooled_ks_target = ks_distance(pooled_speeds, target_scale)
    pooled_ks_measured = ks_distance(pooled_speeds, measured_scale)

    row = {
        "method": method,
        "input_file": str(input_file),
        "frames_used": stable_velocity.shape[0],
        "samples_used": pooled_speeds.size,
        "temperature_mean_K": mean_temperature,
        "temperature_relative_error": (
            mean_temperature - TARGET_TEMPERATURE_K
        ) / TARGET_TEMPERATURE_K,
        "temperature_std_K": temperature_std,
        "temperature_relative_std": temperature_std / mean_temperature,
        "temperature_relative_variance": relative_variance,
        "canonical_relative_variance_2_over_3N": canonical_relative_variance,
        "fluctuation_ratio_Qvar": fluctuation_ratio,
        "speed_mean_nm_ps": float(np.mean(pooled_speeds)),
        "speed_std_nm_ps": float(np.std(pooled_speeds, ddof=1)),
        "speed_KS_target_pooled": pooled_ks_target,
        "speed_KS_measuredT_pooled": pooled_ks_measured,
        "speed_KS_target_frame_mean": float(np.mean(frame_ks_target)),
        "speed_KS_target_frame_std": float(np.std(frame_ks_target, ddof=1)),
    }

    arrays = {
        "pooled_speeds": pooled_speeds,
        "temperatures": temperatures,
        "frame_ks_target": frame_ks_target,
    }
    return row, arrays


def plot_speed_distributions(rows, arrays_by_method):
    scale = theoretical_scale_nm_ps(TARGET_TEMPERATURE_K)
    maximum_speed = max(
        float(np.max(arrays_by_method[m]["pooled_speeds"]))
        for m in METHODS
    )
    x = np.linspace(0.0, 1.05 * maximum_speed, 1000)
    bins = np.linspace(0.0, 1.05 * maximum_speed, N_BINS + 1)

    fig, axis = plt.subplots(figsize=(10, 6.5))
    for row in rows:
        method = str(row["method"])
        axis.hist(
            arrays_by_method[method]["pooled_speeds"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            label=(
                f"{method.capitalize()} "
                f"(KS={float(row['speed_KS_target_pooled']):.4f})"
            ),
        )

    axis.plot(
        x,
        maxwell.pdf(x, loc=0.0, scale=scale),
        linestyle="--",
        linewidth=2.8,
        label="Maxwell theory at 300 K",
    )
    axis.set_title("Speed distributions for different thermostat methods")
    axis.set_xlabel("Speed (nm/ps)")
    axis.set_ylabel("Probability density")
    axis.set_xlim(0.0, 1.05 * maximum_speed)
    axis.set_ylim(bottom=0.0)
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figure1_speed_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temperature_series(rows, arrays_by_method):
    fig, axis = plt.subplots(figsize=(10, 6))
    for row in rows:
        method = str(row["method"])
        series = arrays_by_method[method]["temperatures"]
        time_ps = np.arange(series.size) * DT_PS
        axis.plot(
            time_ps,
            series,
            linewidth=1.0,
            label=(
                f"{method.capitalize()}: "
                f"{float(row['temperature_mean_K']):.1f} ± "
                f"{float(row['temperature_std_K']):.1f} K"
            ),
        )

    axis.axhline(
        TARGET_TEMPERATURE_K,
        linestyle="--",
        linewidth=1.8,
        label="Target 300 K",
    )
    axis.set_title("Instantaneous temperature trajectories")
    axis.set_xlabel("Equilibrated trajectory time (ps)")
    axis.set_ylabel("Temperature (K)")
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figure2_temperature_series.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_mean_temperature(rows):
    labels = [str(row["method"]).capitalize() for row in rows]
    means = [float(row["temperature_mean_K"]) for row in rows]
    stds = [float(row["temperature_std_K"]) for row in rows]

    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.bar(labels, means, yerr=stds, capsize=5)
    axis.axhline(TARGET_TEMPERATURE_K, linestyle="--", label="Target 300 K")
    axis.set_title("Mean temperature with temporal standard deviation")
    axis.set_ylabel("Temperature (K)")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figure3_mean_temperature.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temperature_fluctuations(rows):
    labels = [str(row["method"]).capitalize() for row in rows]
    relative_variances = [
        float(row["temperature_relative_variance"])
        for row in rows
    ]
    theory = 2.0 / (3.0 * N_PARTICLES)

    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.bar(labels, relative_variances)
    axis.axhline(
        theory,
        linestyle="--",
        linewidth=2.0,
        label=r"Canonical $2/(3N)$",
    )
    axis.set_title("Relative temperature variance")
    axis.set_ylabel(r"$\mathrm{Var}(T)/\langle T\rangle^2$")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figure4_temperature_fluctuations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ks_comparison(rows):
    labels = [str(row["method"]).capitalize() for row in rows]
    pooled = [float(row["speed_KS_target_pooled"]) for row in rows]
    frame_means = [float(row["speed_KS_target_frame_mean"]) for row in rows]
    frame_stds = [float(row["speed_KS_target_frame_std"]) for row in rows]

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, axis = plt.subplots(figsize=(9, 5.7))
    axis.bar(x - width / 2, pooled, width=width, label="Pooled KS")
    axis.bar(
        x + width / 2,
        frame_means,
        width=width,
        yerr=frame_stds,
        capsize=4,
        label="Frame-wise KS mean ± SD",
    )
    axis.set_xticks(x, labels)
    axis.set_title("Deviation from the 300 K Maxwell distribution")
    axis.set_ylabel("KS statistic")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figure5_KS_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_outputs(rows):
    csv_file = OUTPUT_DIR / "thermostat_statistics.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    text_file = OUTPUT_DIR / "thermostat_statistics.txt"
    with text_file.open("w", encoding="utf-8") as handle:
        handle.write("Thermostat comparison\n")
        handle.write("=" * 118 + "\n")
        handle.write(
            f"Fixed conditions: T_target={TARGET_TEMPERATURE_K:g} K, "
            f"N={N_PARTICLES}, L=100 nm\n"
        )
        handle.write(
            f"Canonical relative temperature variance 2/(3N) = "
            f"{2.0/(3.0*N_PARTICLES):.8f}\n\n"
        )
        header = (
            f"{'method':<12}"
            f"{'<T>/K':>11}"
            f"{'T err %':>11}"
            f"{'std T/K':>12}"
            f"{'rel var T':>13}"
            f"{'Qvar':>10}"
            f"{'KS pooled':>12}"
            f"{'KS frame':>12}"
            f"{'KS own T':>11}\n"
        )
        handle.write(header)
        handle.write("-" * len(header) + "\n")
        for row in rows:
            handle.write(
                f"{str(row['method']):<12}"
                f"{float(row['temperature_mean_K']):>11.3f}"
                f"{100*float(row['temperature_relative_error']):>11.3f}"
                f"{float(row['temperature_std_K']):>12.3f}"
                f"{float(row['temperature_relative_variance']):>13.6f}"
                f"{float(row['fluctuation_ratio_Qvar']):>10.3f}"
                f"{float(row['speed_KS_target_pooled']):>12.6f}"
                f"{float(row['speed_KS_target_frame_mean']):>12.6f}"
                f"{float(row['speed_KS_measuredT_pooled']):>11.6f}\n"
            )

        handle.write("\nInterpretation:\n")
        handle.write("  Qvar ≈ 1: canonical temperature fluctuations are reproduced.\n")
        handle.write("  Qvar < 1: fluctuations are suppressed.\n")
        handle.write("  NVE is not expected to satisfy the NVT fluctuation criterion.\n")
        handle.write(
            "  KS own T separates distribution-shape error from an average-"
            "temperature offset.\n"
        )


def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        arrays_by_method = {}

        for method in METHODS:
            row, arrays = load_method(method)
            rows.append(row)
            arrays_by_method[method] = arrays
            print(f"Loaded {method}: {INPUT_FILES[method]}")

        plot_speed_distributions(rows, arrays_by_method)
        plot_temperature_series(rows, arrays_by_method)
        plot_mean_temperature(rows)
        plot_temperature_fluctuations(rows)
        plot_ks_comparison(rows)
        write_outputs(rows)

        print("\nThermostat comparison")
        print("-" * 110)
        print(
            f"{'method':<12}{'<T>/K':>10}{'std T':>10}"
            f"{'rel var':>12}{'Qvar':>10}{'KS pooled':>12}"
            f"{'KS frame':>12}{'KS own T':>11}"
        )
        for row in rows:
            print(
                f"{str(row['method']):<12}"
                f"{float(row['temperature_mean_K']):>10.3f}"
                f"{float(row['temperature_std_K']):>10.3f}"
                f"{float(row['temperature_relative_variance']):>12.6f}"
                f"{float(row['fluctuation_ratio_Qvar']):>10.3f}"
                f"{float(row['speed_KS_target_pooled']):>12.6f}"
                f"{float(row['speed_KS_target_frame_mean']):>12.6f}"
                f"{float(row['speed_KS_measuredT_pooled']):>11.6f}"
            )
        print("-" * 110)
        print(f"Results written to: {OUTPUT_DIR}")
        return 0

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())