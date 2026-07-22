#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze the sqrt(T) scaling of Maxwell-Boltzmann speed distributions.

The program reads the equilibrated velocity trajectories for:
    T = 100 K, 300 K, 600 K
    N = 200
    Langevin NVT

It generates:
    1. temperature_speed_distributions.png
       MD speed histograms and theoretical Maxwell curves.

    2. sqrtT_scaling.png
       Left: mean speed versus sqrt(T), testing right shift.
       Right: speed standard deviation versus sqrt(T), testing broadening.

    3. temperature_scaling_statistics.csv
       Numerical comparison of MD and theory.

    4. temperature_scaling_statistics.txt
       Human-readable results and expected/observed ratios.

Run:
    python analyze_temperature_scaling.py
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


# ============================================================
# User settings
# ============================================================

MASS_ARGON_G_MOL = 39.95
EQUILIBRATION_FRACTION = 0.30
N_BINS = 60

OUTPUT_DIR = Path("temperature_MB_results")

# The program tries these filenames in order for each temperature.
FILE_CANDIDATES = {
    100.0: (
        Path("mb_results/mb_T100_N200_langevin_vel.npy"),
        Path("my_simulation_100K_vel.npy"),
        Path("my_simulation_100K_N200_L100_Langevin_vel.npy"),
    ),
    300.0: (
        Path("mb_results/mb_T300_N200_langevin_vel.npy"),
        Path("my_simulation_300K_vel.npy"),
        Path("my_simulation_300K_N200_L100_Langevin_vel.npy"),
    ),
    600.0: (
        Path("mb_results/mb_T600_N200_langevin_vel.npy"),
        Path("my_simulation_600K_vel.npy"),
        Path("my_simulation_600K_N200_L100_Langevin_vel.npy"),
    ),
}


# ============================================================
# Theory
# ============================================================

def sigma_component_theory(
    temperature_k: float,
    mass_g_mol: float,
) -> float:
    """
    Maxwell scale parameter / component standard deviation:

        sigma = sqrt(R T / M)

    Return unit: nm/ps.
    """
    mass_kg_mol = mass_g_mol * 1e-3
    return float(
        np.sqrt(R * temperature_k / mass_kg_mol) * 1e-3
    )


def theoretical_values(
    temperature_k: float,
    mass_g_mol: float,
) -> dict[str, float]:
    sigma = sigma_component_theory(
        temperature_k,
        mass_g_mol,
    )

    return {
        "component_sigma": sigma,
        "mode_speed": np.sqrt(2.0) * sigma,
        "mean_speed": 2.0 * sigma * np.sqrt(2.0 / np.pi),
        "std_speed": sigma * np.sqrt(3.0 - 8.0 / np.pi),
        "rms_speed": np.sqrt(3.0) * sigma,
    }


# ============================================================
# File loading
# ============================================================

def find_input_file(
    temperature_k: float,
) -> Path:
    for candidate in FILE_CANDIDATES[temperature_k]:
        if candidate.is_file():
            return candidate

    candidates_text = "\n".join(
        f"  - {path}"
        for path in FILE_CANDIDATES[temperature_k]
    )

    raise FileNotFoundError(
        f"No velocity file found for T={temperature_k:g} K.\n"
        f"Expected one of:\n{candidates_text}\n"
        "Edit FILE_CANDIDATES at the top of the program "
        "if your filenames are different."
    )


def load_stable_velocity(
    input_file: Path,
) -> tuple[np.ndarray, int]:
    velocity = np.load(
        input_file,
        allow_pickle=False,
    )

    if velocity.ndim != 3 or velocity.shape[2] != 3:
        raise ValueError(
            f"{input_file} has shape {velocity.shape}; "
            "expected (n_frames, n_particles, 3)."
        )

    if not np.all(np.isfinite(velocity)):
        raise ValueError(
            f"{input_file} contains NaN or infinity."
        )

    first_frame = int(
        EQUILIBRATION_FRACTION * velocity.shape[0]
    )

    stable_velocity = velocity[first_frame:]

    if stable_velocity.shape[0] < 2:
        raise ValueError(
            f"Too few equilibrated frames in {input_file}."
        )

    return stable_velocity, first_frame


# ============================================================
# MD statistics
# ============================================================

def temperature_from_velocities(
    stable_velocity: np.ndarray,
    mass_g_mol: float,
) -> tuple[float, float]:
    """
    Calculate the instantaneous temperature exactly as in LJ_gas.py:

        T(t) = 2 K(t) / (3 N R)

    For mass in g/mol and velocity in nm/ps,
    0.5 * mass * v^2 has units kJ/mol.
    """
    n_particles = stable_velocity.shape[1]

    sum_v_squared = np.sum(
        stable_velocity**2,
        axis=(1, 2),
    )

    kinetic_energy_kj_mol = (
        0.5
        * mass_g_mol
        * sum_v_squared
    )

    temperature_frames = (
        2.0
        * kinetic_energy_kj_mol
        * 1e3
        / (3.0 * n_particles * R)
    )

    return (
        float(np.mean(temperature_frames)),
        float(np.std(temperature_frames, ddof=1)),
    )


def analyze_one_temperature(
    target_temperature_k: float,
    input_file: Path,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    stable_velocity, first_frame = load_stable_velocity(
        input_file
    )

    n_particles = stable_velocity.shape[1]

    speeds = np.linalg.norm(
        stable_velocity,
        axis=2,
    ).reshape(-1)

    vx = stable_velocity[:, :, 0].reshape(-1)
    vy = stable_velocity[:, :, 1].reshape(-1)
    vz = stable_velocity[:, :, 2].reshape(-1)

    component_stds = np.array(
        [
            np.std(vx, ddof=1),
            np.std(vy, ddof=1),
            np.std(vz, ddof=1),
        ]
    )

    mean_component_std = float(
        np.mean(component_stds)
    )

    md_mean_speed = float(
        np.mean(speeds)
    )

    md_std_speed = float(
        np.std(speeds, ddof=1)
    )

    # Maxwell maximum-likelihood scale estimate:
    #
    #     a_hat = sqrt(<v^2>/3)
    #
    # This provides a robust estimate of the position of
    # the Maxwell peak:
    #
    #     v_mp,fit = sqrt(2) a_hat
    fitted_scale = float(
        np.sqrt(
            np.mean(speeds**2)
            / 3.0
        )
    )

    fitted_mode_speed = float(
        np.sqrt(2.0)
        * fitted_scale
    )

    temperature_mean, temperature_std = (
        temperature_from_velocities(
            stable_velocity,
            MASS_ARGON_G_MOL,
        )
    )

    theory = theoretical_values(
        target_temperature_k,
        MASS_ARGON_G_MOL,
    )

    # KS distance to the temperature-specific theoretical curve.
    sorted_speed = np.sort(speeds)
    empirical_cdf = np.arange(
        1,
        sorted_speed.size + 1,
    ) / sorted_speed.size

    theoretical_cdf = maxwell.cdf(
        sorted_speed,
        loc=0.0,
        scale=theory["component_sigma"],
    )

    ks_D = float(
        np.max(
            np.abs(
                empirical_cdf
                - theoretical_cdf
            )
        )
    )

    sqrt_temperature = float(
        np.sqrt(target_temperature_k)
    )

    row = {
        "target_temperature_K": target_temperature_k,
        "measured_temperature_K": temperature_mean,
        "temperature_std_K": temperature_std,
        "n_particles": n_particles,
        "frames_used": stable_velocity.shape[0],
        "first_frame_used": first_frame,
        "n_speed_samples": speeds.size,
        "input_file": str(input_file),

        "md_component_sigma_nm_ps": mean_component_std,
        "theory_component_sigma_nm_ps": theory["component_sigma"],
        "component_sigma_relative_error_percent": (
            abs(
                mean_component_std
                - theory["component_sigma"]
            )
            / theory["component_sigma"]
            * 100.0
        ),

        "md_mean_speed_nm_ps": md_mean_speed,
        "theory_mean_speed_nm_ps": theory["mean_speed"],
        "mean_speed_relative_error_percent": (
            abs(
                md_mean_speed
                - theory["mean_speed"]
            )
            / theory["mean_speed"]
            * 100.0
        ),

        "md_speed_std_nm_ps": md_std_speed,
        "theory_speed_std_nm_ps": theory["std_speed"],
        "speed_std_relative_error_percent": (
            abs(
                md_std_speed
                - theory["std_speed"]
            )
            / theory["std_speed"]
            * 100.0
        ),

        "md_fitted_mode_speed_nm_ps": fitted_mode_speed,
        "theory_mode_speed_nm_ps": theory["mode_speed"],
        "mode_relative_error_percent": (
            abs(
                fitted_mode_speed
                - theory["mode_speed"]
            )
            / theory["mode_speed"]
            * 100.0
        ),

        "theory_rms_speed_nm_ps": theory["rms_speed"],
        "speed_KS_D": ks_D,

        # If sqrt(T) scaling is correct, these normalized
        # quantities should be approximately constant.
        "md_mean_speed_over_sqrtT": (
            md_mean_speed
            / sqrt_temperature
        ),
        "theory_mean_speed_over_sqrtT": (
            theory["mean_speed"]
            / sqrt_temperature
        ),
        "md_speed_std_over_sqrtT": (
            md_std_speed
            / sqrt_temperature
        ),
        "theory_speed_std_over_sqrtT": (
            theory["std_speed"]
            / sqrt_temperature
        ),
        "md_mode_over_sqrtT": (
            fitted_mode_speed
            / sqrt_temperature
        ),
        "theory_mode_over_sqrtT": (
            theory["mode_speed"]
            / sqrt_temperature
        ),
    }

    return row, speeds


# ============================================================
# Scaling fits
# ============================================================

def fit_through_origin(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """
    Fit y = slope * x through the origin.

    Returns:
        slope, R^2
    """
    slope = float(
        np.dot(x, y)
        / np.dot(x, x)
    )

    prediction = slope * x

    ss_res = float(
        np.sum(
            (y - prediction)**2
        )
    )

    ss_tot = float(
        np.sum(
            (y - np.mean(y))**2
        )
    )

    if ss_tot == 0.0:
        r_squared = float("nan")
    else:
        r_squared = 1.0 - ss_res / ss_tot

    return slope, r_squared


# ============================================================
# Plotting
# ============================================================

def plot_speed_distributions(
    rows: list[dict[str, float | int | str]],
    speeds_by_temperature: dict[float, np.ndarray],
    output_file: Path,
) -> None:
    maximum_speed = max(
        float(np.max(values))
        for values in speeds_by_temperature.values()
    )

    speed_grid = np.linspace(
        0.0,
        1.05 * maximum_speed,
        1000,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    for row in rows:
        temperature = float(
            row["target_temperature_K"]
        )

        speeds = speeds_by_temperature[
            temperature
        ]

        axis.hist(
            speeds,
            bins=N_BINS,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=f"MD {temperature:g} K",
        )

        sigma = float(
            row[
                "theory_component_sigma_nm_ps"
            ]
        )

        axis.plot(
            speed_grid,
            maxwell.pdf(
                speed_grid,
                loc=0.0,
                scale=sigma,
            ),
            linewidth=2.0,
            label=f"Theory {temperature:g} K",
        )

        # Mark the theoretical most probable speed.
        axis.axvline(
            float(
                row[
                    "theory_mode_speed_nm_ps"
                ]
            ),
            linestyle=":",
            linewidth=1.0,
        )

    axis.set_title(
        "Maxwell-Boltzmann speed distributions "
        "at different temperatures"
    )
    axis.set_xlabel(
        "Speed (nm/ps)"
    )
    axis.set_ylabel(
        "Probability density"
    )
    axis.set_xlim(
        0.0,
        1.05 * maximum_speed,
    )
    axis.set_ylim(
        bottom=0.0
    )
    axis.grid(
        linestyle="--",
        alpha=0.35,
    )
    axis.legend(
        ncol=2,
        fontsize=9,
    )

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_sqrt_temperature_scaling(
    rows: list[dict[str, float | int | str]],
    output_file: Path,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    temperatures = np.array(
        [
            float(row["target_temperature_K"])
            for row in rows
        ]
    )

    sqrt_temperatures = np.sqrt(
        temperatures
    )

    md_means = np.array(
        [
            float(row["md_mean_speed_nm_ps"])
            for row in rows
        ]
    )

    theory_means = np.array(
        [
            float(row["theory_mean_speed_nm_ps"])
            for row in rows
        ]
    )

    md_stds = np.array(
        [
            float(row["md_speed_std_nm_ps"])
            for row in rows
        ]
    )

    theory_stds = np.array(
        [
            float(row["theory_speed_std_nm_ps"])
            for row in rows
        ]
    )

    mean_slope, mean_r_squared = (
        fit_through_origin(
            sqrt_temperatures,
            md_means,
        )
    )

    std_slope, std_r_squared = (
        fit_through_origin(
            sqrt_temperatures,
            md_stds,
        )
    )

    x_line = np.linspace(
        0.0,
        1.05 * np.max(sqrt_temperatures),
        300,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.2),
    )

    # Right shift: mean speed.
    axes[0].scatter(
        sqrt_temperatures,
        md_means,
        s=70,
        label="MD mean speed",
    )

    axes[0].plot(
        sqrt_temperatures,
        theory_means,
        linestyle="--",
        linewidth=2.0,
        label="Theory",
    )

    axes[0].plot(
        x_line,
        mean_slope * x_line,
        linewidth=1.5,
        label=(
            f"MD fit: y={mean_slope:.5f}"
            f"$\\sqrt{{T}}$, "
            f"$R^2={mean_r_squared:.5f}$"
        ),
    )

    for x_value, y_value, temperature in zip(
        sqrt_temperatures,
        md_means,
        temperatures,
    ):
        axes[0].annotate(
            f"{temperature:g} K",
            (x_value, y_value),
            xytext=(5, 6),
            textcoords="offset points",
        )

    axes[0].set_title(
        "Right shift: mean speed"
    )
    axes[0].set_xlabel(
        r"$\sqrt{T}$ ($\sqrt{\mathrm{K}}$)"
    )
    axes[0].set_ylabel(
        "Mean speed (nm/ps)"
    )
    axes[0].set_xlim(
        left=0.0
    )
    axes[0].set_ylim(
        bottom=0.0
    )
    axes[0].grid(
        linestyle="--",
        alpha=0.35,
    )
    axes[0].legend(
        fontsize=9,
    )

    # Broadening: speed standard deviation.
    axes[1].scatter(
        sqrt_temperatures,
        md_stds,
        s=70,
        label="MD speed std",
    )

    axes[1].plot(
        sqrt_temperatures,
        theory_stds,
        linestyle="--",
        linewidth=2.0,
        label="Theory",
    )

    axes[1].plot(
        x_line,
        std_slope * x_line,
        linewidth=1.5,
        label=(
            f"MD fit: y={std_slope:.5f}"
            f"$\\sqrt{{T}}$, "
            f"$R^2={std_r_squared:.5f}$"
        ),
    )

    for x_value, y_value, temperature in zip(
        sqrt_temperatures,
        md_stds,
        temperatures,
    ):
        axes[1].annotate(
            f"{temperature:g} K",
            (x_value, y_value),
            xytext=(5, 6),
            textcoords="offset points",
        )

    axes[1].set_title(
        "Broadening: speed standard deviation"
    )
    axes[1].set_xlabel(
        r"$\sqrt{T}$ ($\sqrt{\mathrm{K}}$)"
    )
    axes[1].set_ylabel(
        "Speed standard deviation (nm/ps)"
    )
    axes[1].set_xlim(
        left=0.0
    )
    axes[1].set_ylim(
        bottom=0.0
    )
    axes[1].grid(
        linestyle="--",
        alpha=0.35,
    )
    axes[1].legend(
        fontsize=9,
    )

    figure.suptitle(
        r"Test of the expected $\sqrt{T}$ scaling",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.94)
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return (
        mean_slope,
        mean_r_squared,
        std_slope,
        std_r_squared,
    )


# ============================================================
# Output files
# ============================================================

def write_csv(
    rows: list[dict[str, float | int | str]],
    output_file: Path,
) -> None:
    fieldnames = list(
        rows[0].keys()
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(
    rows: list[dict[str, float | int | str]],
    mean_slope: float,
    mean_r_squared: float,
    std_slope: float,
    std_r_squared: float,
    output_file: Path,
) -> None:
    temperatures = [
        float(row["target_temperature_K"])
        for row in rows
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Temperature scaling of the "
            "Maxwell-Boltzmann distribution\n"
        )
        handle.write(
            "=" * 78 + "\n\n"
        )

        handle.write(
            "A quantity follows sqrt(T) scaling if "
            "quantity/sqrt(T) is approximately constant.\n\n"
        )

        header = (
            f"{'T/K':>8}"
            f"{'T_MD/K':>12}"
            f"{'<v> MD':>13}"
            f"{'<v> theory':>14}"
            f"{'std(v) MD':>14}"
            f"{'std theory':>14}"
            f"{'vmp fit':>13}"
            f"{'vmp theory':>14}"
            f"{'KS D':>10}\n"
        )

        handle.write(header)
        handle.write(
            "-" * len(header) + "\n"
        )

        for row in rows:
            handle.write(
                f"{float(row['target_temperature_K']):>8.1f}"
                f"{float(row['measured_temperature_K']):>12.3f}"
                f"{float(row['md_mean_speed_nm_ps']):>13.6f}"
                f"{float(row['theory_mean_speed_nm_ps']):>14.6f}"
                f"{float(row['md_speed_std_nm_ps']):>14.6f}"
                f"{float(row['theory_speed_std_nm_ps']):>14.6f}"
                f"{float(row['md_fitted_mode_speed_nm_ps']):>13.6f}"
                f"{float(row['theory_mode_speed_nm_ps']):>14.6f}"
                f"{float(row['speed_KS_D']):>10.6f}\n"
            )

        handle.write("\n")
        handle.write(
            "Fit through origin:\n"
        )
        handle.write(
            f"  MD mean speed = "
            f"{mean_slope:.8f} * sqrt(T), "
            f"R^2 = {mean_r_squared:.8f}\n"
        )
        handle.write(
            f"  MD speed std  = "
            f"{std_slope:.8f} * sqrt(T), "
            f"R^2 = {std_r_squared:.8f}\n\n"
        )

        handle.write(
            "Expected temperature ratios:\n"
        )

        for i in range(
            len(temperatures)
        ):
            for j in range(
                i + 1,
                len(temperatures),
            ):
                t1 = temperatures[i]
                t2 = temperatures[j]

                expected_ratio = np.sqrt(
                    t2 / t1
                )

                row1 = rows[i]
                row2 = rows[j]

                mean_ratio = (
                    float(
                        row2[
                            "md_mean_speed_nm_ps"
                        ]
                    )
                    / float(
                        row1[
                            "md_mean_speed_nm_ps"
                        ]
                    )
                )

                std_ratio = (
                    float(
                        row2[
                            "md_speed_std_nm_ps"
                        ]
                    )
                    / float(
                        row1[
                            "md_speed_std_nm_ps"
                        ]
                    )
                )

                mode_ratio = (
                    float(
                        row2[
                            "md_fitted_mode_speed_nm_ps"
                        ]
                    )
                    / float(
                        row1[
                            "md_fitted_mode_speed_nm_ps"
                        ]
                    )
                )

                handle.write(
                    f"  {t2:g} K / {t1:g} K:\n"
                    f"    expected sqrt(T2/T1) = "
                    f"{expected_ratio:.6f}\n"
                    f"    MD mean-speed ratio  = "
                    f"{mean_ratio:.6f}\n"
                    f"    MD std-speed ratio   = "
                    f"{std_ratio:.6f}\n"
                    f"    MD fitted-mode ratio = "
                    f"{mode_ratio:.6f}\n"
                )


# ============================================================
# Main
# ============================================================

def main() -> int:
    try:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []
        speeds_by_temperature = {}

        for temperature in sorted(
            FILE_CANDIDATES
        ):
            input_file = find_input_file(
                temperature
            )

            row, speeds = analyze_one_temperature(
                temperature,
                input_file,
            )

            rows.append(row)
            speeds_by_temperature[
                temperature
            ] = speeds

            print(
                f"Loaded T={temperature:g} K: "
                f"{input_file}"
            )

        plot_speed_distributions(
            rows,
            speeds_by_temperature,
            OUTPUT_DIR
            / "temperature_speed_distributions.png",
        )

        (
            mean_slope,
            mean_r_squared,
            std_slope,
            std_r_squared,
        ) = plot_sqrt_temperature_scaling(
            rows,
            OUTPUT_DIR
            / "sqrtT_scaling.png",
        )

        write_csv(
            rows,
            OUTPUT_DIR
            / "temperature_scaling_statistics.csv",
        )

        write_text_summary(
            rows,
            mean_slope,
            mean_r_squared,
            std_slope,
            std_r_squared,
            OUTPUT_DIR
            / "temperature_scaling_statistics.txt",
        )

        print(
            "\nTemperature scaling results"
        )
        print(
            "-" * 92
        )
        print(
            f"{'T/K':>7}"
            f"{'T_MD/K':>11}"
            f"{'<v> MD':>13}"
            f"{'<v> theory':>14}"
            f"{'std MD':>12}"
            f"{'std theory':>13}"
            f"{'vmp fit':>12}"
            f"{'vmp theory':>13}"
        )

        for row in rows:
            print(
                f"{float(row['target_temperature_K']):>7.1f}"
                f"{float(row['measured_temperature_K']):>11.3f}"
                f"{float(row['md_mean_speed_nm_ps']):>13.6f}"
                f"{float(row['theory_mean_speed_nm_ps']):>14.6f}"
                f"{float(row['md_speed_std_nm_ps']):>12.6f}"
                f"{float(row['theory_speed_std_nm_ps']):>13.6f}"
                f"{float(row['md_fitted_mode_speed_nm_ps']):>12.6f}"
                f"{float(row['theory_mode_speed_nm_ps']):>13.6f}"
            )

        print(
            "-" * 92
        )
        print(
            f"Mean-speed fit R^2: "
            f"{mean_r_squared:.8f}"
        )
        print(
            f"Speed-std fit R^2:  "
            f"{std_r_squared:.8f}"
        )

        print(
            f"\nResults saved in: {OUTPUT_DIR}"
        )
        print(
            "  temperature_speed_distributions.png"
        )
        print(
            "  sqrtT_scaling.png"
        )
        print(
            "  temperature_scaling_statistics.csv"
        )
        print(
            "  temperature_scaling_statistics.txt"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )