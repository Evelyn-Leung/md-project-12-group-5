#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Particle-number dependence of Maxwell-Boltzmann sampling.

Fixed conditions:
    T = 300 K
    L = 100 nm
    Thermostat = Langevin
    N = 20, 50, 100, 200

Input arrays must have shape:
    (n_frames, n_particles, 3)

Outputs:
    particle_number_MB_results/
        figure1_particle_number_distributions.png
        figure2_small_system_fluctuations.png
        particle_number_statistics.csv
        particle_number_statistics.txt

Figure 1:
    Four MD speed histograms at fixed T and one common theoretical
    Maxwell curve. The theoretical curve does not depend on N.

Figure 2:
    Left: instantaneous temperature fluctuations T(t)/T_target.
    Right: mean frame-wise speed KS statistic with error bars.

Run:
    python analyze_particle_number_effect.py
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

TEMPERATURE_K = 300.0
MASS_ARGON_G_MOL = 39.95
PARTICLE_NUMBERS = (20, 50, 100, 200)

DT_PS = 0.1
EQUILIBRATION_FRACTION = 0.30
N_BINS = 60

OUTPUT_DIR = Path("particle_number_MB_results")

# The script tries each filename in order.
FILE_CANDIDATES = {
    20: (
        Path("mb_results_fixed_L100/mb_T300_N20_L100_langevin_vel.npy"),
        Path("my_simulation_300K_N20_L100_Langevin_vel.npy"),
    ),
    50: (
        Path("mb_results_fixed_L100/mb_T300_N50_L100_langevin_vel.npy"),
        Path("my_simulation_300K_N50_L100_Langevin_vel.npy"),
    ),
    100: (
        Path("mb_results_fixed_L100/mb_T300_N100_L100_langevin_vel.npy"),
        Path("my_simulation_300K_N100_L100_Langevin_vel.npy"),
    ),
    200: (
        Path("mb_results_fixed_L100/mb_T300_N200_L100_langevin_vel.npy"),
        Path("my_simulation_300K_N200_L100_Langevin_vel.npy"),
    ),
}


# ============================================================
# Theory
# ============================================================

def component_sigma_theory_nm_ps() -> float:
    """
    One-dimensional MB standard deviation:

        sigma = sqrt(R T / M)

    converted from m/s to nm/ps.
    """
    mass_kg_mol = MASS_ARGON_G_MOL * 1e-3
    return float(
        np.sqrt(R * TEMPERATURE_K / mass_kg_mol) * 1e-3
    )


SIGMA_THEORY = component_sigma_theory_nm_ps()

MEAN_SPEED_THEORY = (
    2.0 * SIGMA_THEORY * np.sqrt(2.0 / np.pi)
)

STD_SPEED_THEORY = (
    SIGMA_THEORY * np.sqrt(3.0 - 8.0 / np.pi)
)

MODE_SPEED_THEORY = np.sqrt(2.0) * SIGMA_THEORY


def theoretical_relative_temperature_std(n_particles: int) -> float:
    """
    Canonical ideal-gas prediction:

        std(T) / <T> = sqrt(2 / (3N))

    This assumes 3N kinetic degrees of freedom, matching LJ_gas.py.
    """
    return float(np.sqrt(2.0 / (3.0 * n_particles)))


# ============================================================
# File loading
# ============================================================

def find_input_file(n_particles: int) -> Path:
    for candidate in FILE_CANDIDATES[n_particles]:
        if candidate.is_file():
            return candidate

    choices = "\n".join(
        f"  - {path}" for path in FILE_CANDIDATES[n_particles]
    )

    raise FileNotFoundError(
        f"No velocity file found for N={n_particles}.\n"
        f"Expected one of:\n{choices}\n"
        "Edit FILE_CANDIDATES at the top of the script if needed."
    )


def load_velocity(
    input_file: Path,
    expected_n: int,
) -> tuple[np.ndarray, int]:
    velocity = np.load(input_file, allow_pickle=False)

    if velocity.ndim != 3 or velocity.shape[2] != 3:
        raise ValueError(
            f"{input_file} has shape {velocity.shape}; "
            "expected (n_frames, n_particles, 3)."
        )

    if velocity.shape[1] != expected_n:
        raise ValueError(
            f"{input_file} contains N={velocity.shape[1]}, "
            f"but N={expected_n} was expected."
        )

    if not np.all(np.isfinite(velocity)):
        raise ValueError(f"{input_file} contains NaN or infinity.")

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
# Statistics
# ============================================================

def instantaneous_temperature(
    stable_velocity: np.ndarray,
) -> np.ndarray:
    """
    Same 3N-degree-of-freedom convention as LJ_gas.py.

    For velocity in nm/ps and molar mass in g/mol:
        K [kJ/mol] = 1/2 M sum(v^2)

        T = 2 K * 1000 / (3 N R)
    """
    n_particles = stable_velocity.shape[1]

    sum_v2_per_frame = np.sum(
        stable_velocity**2,
        axis=(1, 2),
    )

    kinetic_energy_kj_mol = (
        0.5
        * MASS_ARGON_G_MOL
        * sum_v2_per_frame
    )

    return (
        2.0
        * kinetic_energy_kj_mol
        * 1e3
        / (3.0 * n_particles * R)
    )


def empirical_ks_distance(
    sample: np.ndarray,
    scale: float,
) -> float:
    """
    Descriptive one-sample KS distance:

        D = sup_v |F_MD(v) - F_theory(v)|
    """
    values = np.sort(sample)

    n_values = values.size

    cdf_upper = np.arange(
        1,
        n_values + 1,
        dtype=float,
    ) / n_values

    cdf_lower = np.arange(
        0,
        n_values,
        dtype=float,
    ) / n_values

    theory_cdf = maxwell.cdf(
        values,
        loc=0.0,
        scale=scale,
    )

    return float(
        max(
            np.max(np.abs(cdf_upper - theory_cdf)),
            np.max(np.abs(cdf_lower - theory_cdf)),
        )
    )


def framewise_ks_statistics(
    stable_velocity: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Calculate one speed KS distance per equilibrated frame.

    Each frame contains N speeds, so this retains the finite-system
    effect instead of hiding it by pooling all frames.
    """
    speeds_by_frame = np.linalg.norm(
        stable_velocity,
        axis=2,
    )

    ks_values = np.array(
        [
            empirical_ks_distance(frame_speeds, SIGMA_THEORY)
            for frame_speeds in speeds_by_frame
        ],
        dtype=float,
    )

    return (
        ks_values,
        float(np.mean(ks_values)),
        float(np.std(ks_values, ddof=1)),
    )


def analyze_system(
    n_particles: int,
    input_file: Path,
) -> tuple[
    dict[str, float | int | str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    stable_velocity, first_frame = load_velocity(
        input_file,
        n_particles,
    )

    speeds_by_frame = np.linalg.norm(
        stable_velocity,
        axis=2,
    )
    pooled_speeds = speeds_by_frame.reshape(-1)

    temperatures = instantaneous_temperature(
        stable_velocity
    )

    ks_frames, ks_mean, ks_std = (
        framewise_ks_statistics(
            stable_velocity
        )
    )

    pooled_ks = empirical_ks_distance(
        pooled_speeds,
        SIGMA_THEORY,
    )

    measured_temperature = float(
        np.mean(temperatures)
    )

    temperature_std = float(
        np.std(temperatures, ddof=1)
    )

    relative_temperature_std = (
        temperature_std
        / measured_temperature
    )

    row = {
        "n_particles": n_particles,
        "input_file": str(input_file),
        "frames_used": stable_velocity.shape[0],
        "first_frame_used": first_frame,
        "n_pooled_speed_samples": pooled_speeds.size,

        "temperature_mean_K": measured_temperature,
        "temperature_std_K": temperature_std,
        "temperature_relative_std": relative_temperature_std,
        "temperature_relative_std_theory": (
            theoretical_relative_temperature_std(
                n_particles
            )
        ),
        "temperature_fluctuation_ratio_MD_over_theory": (
            relative_temperature_std
            / theoretical_relative_temperature_std(
                n_particles
            )
        ),

        "mean_speed_MD_nm_ps": float(
            np.mean(pooled_speeds)
        ),
        "mean_speed_theory_nm_ps": float(
            MEAN_SPEED_THEORY
        ),
        "mean_speed_relative_error_percent": float(
            abs(
                np.mean(pooled_speeds)
                - MEAN_SPEED_THEORY
            )
            / MEAN_SPEED_THEORY
            * 100.0
        ),

        "speed_std_MD_nm_ps": float(
            np.std(pooled_speeds, ddof=1)
        ),
        "speed_std_theory_nm_ps": float(
            STD_SPEED_THEORY
        ),
        "speed_std_relative_error_percent": float(
            abs(
                np.std(pooled_speeds, ddof=1)
                - STD_SPEED_THEORY
            )
            / STD_SPEED_THEORY
            * 100.0
        ),

        "pooled_speed_KS_D": pooled_ks,
        "framewise_speed_KS_mean": ks_mean,
        "framewise_speed_KS_std": ks_std,

        # Useful scaling diagnostic:
        # if finite-sample KS ~ C/sqrt(N), then KS*sqrt(N)
        # should be approximately constant.
        "framewise_KS_times_sqrtN": (
            ks_mean * np.sqrt(n_particles)
        ),
    }

    return (
        row,
        pooled_speeds,
        temperatures,
        ks_frames,
    )


# ============================================================
# Plotting
# ============================================================

def plot_particle_number_distributions(
    rows: list[dict[str, float | int | str]],
    speeds_by_n: dict[int, np.ndarray],
    output_file: Path,
) -> None:
    maximum_speed = max(
        float(np.max(speeds))
        for speeds in speeds_by_n.values()
    )

    speed_grid = np.linspace(
        0.0,
        1.05 * maximum_speed,
        1000,
    )

    common_bins = np.linspace(
        0.0,
        1.05 * maximum_speed,
        N_BINS + 1,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    for row in rows:
        n_particles = int(
            row["n_particles"]
        )

        axis.hist(
            speeds_by_n[n_particles],
            bins=common_bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=(
                f"MD N={n_particles}, "
                f"KS={float(row['pooled_speed_KS_D']):.4f}"
            ),
        )

    # Only one theoretical curve is required because T and M are fixed.
    axis.plot(
        speed_grid,
        maxwell.pdf(
            speed_grid,
            loc=0.0,
            scale=SIGMA_THEORY,
        ),
        linestyle="--",
        linewidth=2.8,
        label=(
            "Common Maxwell theory "
            f"(T={TEMPERATURE_K:g} K)"
        ),
    )

    axis.axvline(
        MODE_SPEED_THEORY,
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Theoretical mode = "
            f"{MODE_SPEED_THEORY:.4f} nm/ps"
        ),
    )

    axis.set_title(
        "Figure 1. Effect of particle number on the "
        "equilibrium speed distribution\n"
        "The theoretical Maxwell curve is identical for all N "
        "because T and particle mass are fixed"
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
        fontsize=9,
        ncol=2,
    )

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_small_system_fluctuations(
    rows: list[dict[str, float | int | str]],
    temperatures_by_n: dict[int, np.ndarray],
    output_file: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
    )

    # --------------------------------------------------------
    # Left: instantaneous temperature fluctuations
    # --------------------------------------------------------
    for row in rows:
        n_particles = int(
            row["n_particles"]
        )

        temperature_series = (
            temperatures_by_n[n_particles]
        )

        time_ps = (
            np.arange(temperature_series.size)
            * DT_PS
        )

        axes[0].plot(
            time_ps,
            temperature_series / TEMPERATURE_K,
            linewidth=1.0,
            label=(
                f"N={n_particles}, "
                f"rel. std="
                f"{float(row['temperature_relative_std']):.3f}"
            ),
        )

    axes[0].axhline(
        1.0,
        linestyle="--",
        linewidth=1.7,
        label="Target temperature",
    )

    axes[0].set_title(
        "Instantaneous temperature fluctuations"
    )
    axes[0].set_xlabel(
        "Equilibrated trajectory time (ps)"
    )
    axes[0].set_ylabel(
        r"$T(t)/T_{\mathrm{target}}$"
    )
    axes[0].grid(
        linestyle="--",
        alpha=0.35,
    )
    axes[0].legend(
        fontsize=8,
    )

    # --------------------------------------------------------
    # Right: frame-wise KS deviation versus N
    # --------------------------------------------------------
    particle_numbers = np.array(
        [
            int(row["n_particles"])
            for row in rows
        ],
        dtype=float,
    )

    ks_means = np.array(
        [
            float(
                row[
                    "framewise_speed_KS_mean"
                ]
            )
            for row in rows
        ]
    )

    ks_stds = np.array(
        [
            float(
                row[
                    "framewise_speed_KS_std"
                ]
            )
            for row in rows
        ]
    )

    axes[1].errorbar(
        particle_numbers,
        ks_means,
        yerr=ks_stds,
        marker="o",
        capsize=5,
        linewidth=1.8,
        label="Mean frame-wise KS ± SD",
    )

    # Reference C/sqrt(N), normalized by least-squares fit.
    inv_sqrt_n = 1.0 / np.sqrt(
        particle_numbers
    )

    reference_coefficient = float(
        np.dot(inv_sqrt_n, ks_means)
        / np.dot(inv_sqrt_n, inv_sqrt_n)
    )

    n_grid = np.linspace(
        np.min(particle_numbers),
        np.max(particle_numbers),
        500,
    )

    axes[1].plot(
        n_grid,
        reference_coefficient / np.sqrt(n_grid),
        linestyle="--",
        linewidth=2.0,
        label=(
            r"Reference fit $D=C/\sqrt{N}$"
        ),
    )

    axes[1].set_title(
        "Finite-system deviation from Maxwell theory"
    )
    axes[1].set_xlabel(
        "Number of particles N"
    )
    axes[1].set_ylabel(
        "Frame-wise speed KS statistic"
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
        "Figure 2. Small systems exhibit larger instantaneous "
        "fluctuations and statistical deviations",
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


# ============================================================
# Output tables
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
    output_file: Path,
) -> None:
    with output_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Particle-number dependence of "
            "Maxwell-Boltzmann sampling\n"
        )
        handle.write(
            "=" * 100 + "\n"
        )
        handle.write(
            f"Fixed conditions: T={TEMPERATURE_K:g} K, "
            "L=100 nm, Langevin NVT\n"
        )
        handle.write(
            f"Theoretical Maxwell scale sigma = "
            f"{SIGMA_THEORY:.8f} nm/ps\n"
        )
        handle.write(
            f"Theoretical mean speed = "
            f"{MEAN_SPEED_THEORY:.8f} nm/ps\n"
        )
        handle.write(
            f"Theoretical speed std = "
            f"{STD_SPEED_THEORY:.8f} nm/ps\n"
        )
        handle.write(
            f"Theoretical mode speed = "
            f"{MODE_SPEED_THEORY:.8f} nm/ps\n\n"
        )

        header = (
            f"{'N':>6}"
            f"{'<T>/K':>12}"
            f"{'rel std T':>14}"
            f"{'theory':>12}"
            f"{'<v> err %':>13}"
            f"{'std(v) err %':>15}"
            f"{'pooled KS':>12}"
            f"{'frame KS':>14}"
            f"{'KS SD':>12}"
            f"{'KS*sqrtN':>13}\n"
        )

        handle.write(header)
        handle.write(
            "-" * len(header) + "\n"
        )

        for row in rows:
            handle.write(
                f"{int(row['n_particles']):>6d}"
                f"{float(row['temperature_mean_K']):>12.3f}"
                f"{float(row['temperature_relative_std']):>14.6f}"
                f"{float(row['temperature_relative_std_theory']):>12.6f}"
                f"{float(row['mean_speed_relative_error_percent']):>13.4f}"
                f"{float(row['speed_std_relative_error_percent']):>15.4f}"
                f"{float(row['pooled_speed_KS_D']):>12.6f}"
                f"{float(row['framewise_speed_KS_mean']):>14.6f}"
                f"{float(row['framewise_speed_KS_std']):>12.6f}"
                f"{float(row['framewise_KS_times_sqrtN']):>13.6f}\n"
            )

        handle.write("\nInterpretation\n")
        handle.write("-" * 100 + "\n")
        handle.write(
            "1. At fixed T and particle mass, the theoretical "
            "Maxwell speed curve is independent of N.\n"
        )
        handle.write(
            "2. The pooled histograms may all look smooth because many "
            "correlated frames are combined.\n"
        )
        handle.write(
            "3. Frame-wise KS uses only N speeds per frame and therefore "
            "retains the finite-system sampling effect.\n"
        )
        handle.write(
            "4. For canonical kinetic-energy fluctuations, "
            "std(T)/<T> is expected to scale approximately as "
            "sqrt(2/(3N)).\n"
        )
        handle.write(
            "5. For ordinary finite sampling, a distribution-distance "
            "measure such as KS is expected to decrease approximately "
            "as 1/sqrt(N).\n"
        )
        handle.write(
            "6. Adjacent MD frames are time-correlated, so KS is used "
            "as a descriptive distance, not as a formal p-value test.\n"
        )


def print_summary(
    rows: list[dict[str, float | int | str]],
) -> None:
    print("\nParticle-number results")
    print("-" * 112)
    print(
        f"{'N':>6}"
        f"{'<T>/K':>11}"
        f"{'rel std T':>13}"
        f"{'T theory':>12}"
        f"{'<v> err %':>13}"
        f"{'std err %':>12}"
        f"{'pooled KS':>12}"
        f"{'frame KS':>12}"
        f"{'KS SD':>11}"
    )

    for row in rows:
        print(
            f"{int(row['n_particles']):>6d}"
            f"{float(row['temperature_mean_K']):>11.3f}"
            f"{float(row['temperature_relative_std']):>13.5f}"
            f"{float(row['temperature_relative_std_theory']):>12.5f}"
            f"{float(row['mean_speed_relative_error_percent']):>13.4f}"
            f"{float(row['speed_std_relative_error_percent']):>12.4f}"
            f"{float(row['pooled_speed_KS_D']):>12.5f}"
            f"{float(row['framewise_speed_KS_mean']):>12.5f}"
            f"{float(row['framewise_speed_KS_std']):>11.5f}"
        )

    print("-" * 112)


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
        speeds_by_n = {}
        temperatures_by_n = {}
        ks_by_n = {}

        for n_particles in PARTICLE_NUMBERS:
            input_file = find_input_file(
                n_particles
            )

            (
                row,
                pooled_speeds,
                temperatures,
                ks_frames,
            ) = analyze_system(
                n_particles,
                input_file,
            )

            rows.append(row)
            speeds_by_n[n_particles] = pooled_speeds
            temperatures_by_n[n_particles] = temperatures
            ks_by_n[n_particles] = ks_frames

            print(
                f"Loaded N={n_particles}: "
                f"{input_file}"
            )

        plot_particle_number_distributions(
            rows,
            speeds_by_n,
            OUTPUT_DIR
            / "figure1_particle_number_distributions.png",
        )

        plot_small_system_fluctuations(
            rows,
            temperatures_by_n,
            OUTPUT_DIR
            / "figure2_small_system_fluctuations.png",
        )

        write_csv(
            rows,
            OUTPUT_DIR
            / "particle_number_statistics.csv",
        )

        write_text_summary(
            rows,
            OUTPUT_DIR
            / "particle_number_statistics.txt",
        )

        print_summary(rows)

        print(
            f"\nResults saved in: {OUTPUT_DIR}"
        )
        print(
            "  figure1_particle_number_distributions.png"
        )
        print(
            "  figure2_small_system_fluctuations.png"
        )
        print(
            "  particle_number_statistics.csv"
        )
        print(
            "  particle_number_statistics.txt"
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