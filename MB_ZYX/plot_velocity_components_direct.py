#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read velocity trajectories for N = 20, 50, 100 and plot vx, vy, vz
histograms against the theoretical one-dimensional Maxwell-Boltzmann
Gaussian distribution.

Expected directory structure:

MB/
├── plot_velocity_components_direct.py
└── mb_results_fixed_L100/
    ├── mb_T300_N20_L100_langevin_vel.npy
    ├── mb_T300_N50_L100_langevin_vel.npy
    └── mb_T300_N100_L100_langevin_vel.npy

Run:
    python plot_velocity_components_direct.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R


# ============================================================
# Settings
# ============================================================

TEMPERATURE = 300.0
PARTICLE_NUMBERS = (20, 50, 100)
BOX_LENGTH = 100

MASS_ARGON_G_MOL = 39.95
MASS_ARGON_KG_MOL = MASS_ARGON_G_MOL * 1e-3

EQUILIBRATION_FRACTION = 0.30
N_BINS = 50

INPUT_DIR = Path("mb_results_fixed_L100")
OUTPUT_DIR = Path("mb_component_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Theoretical 1D Maxwell-Boltzmann distribution
# ============================================================

# Each Cartesian velocity component follows a Gaussian distribution:
#
# f(v_i) = 1/(sigma*sqrt(2*pi)) * exp[-v_i^2/(2*sigma^2)]
#
# sigma = sqrt(R*T/M)

sigma_m_s = np.sqrt(R * TEMPERATURE / MASS_ARGON_KG_MOL)
sigma_nm_ps = sigma_m_s * 1e-3


def theoretical_component_pdf(velocity_axis):
    """Return the theoretical Gaussian PDF for vx, vy, or vz."""
    return (
        np.exp(-(velocity_axis ** 2) / (2.0 * sigma_nm_ps ** 2))
        / (sigma_nm_ps * np.sqrt(2.0 * np.pi))
    )


# ============================================================
# Plot one particle-number case
# ============================================================


def plot_case(n_particles):
    input_file = INPUT_DIR / (
        f"mb_T{int(TEMPERATURE)}_N{n_particles}_"
        f"L{BOX_LENGTH}_langevin_vel.npy"
    )

    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    velocity = np.load(input_file)

    if velocity.ndim != 3 or velocity.shape[1] != n_particles or velocity.shape[2] != 3:
        raise ValueError(
            f"Unexpected array shape in {input_file}: {velocity.shape}. "
            f"Expected (n_frames, {n_particles}, 3)."
        )

    # Remove the first 30% of frames as equilibration.
    first_frame = int(EQUILIBRATION_FRACTION * velocity.shape[0])
    stable_velocity = velocity[first_frame:]

    components = {
        "vx": stable_velocity[:, :, 0].reshape(-1),
        "vy": stable_velocity[:, :, 1].reshape(-1),
        "vz": stable_velocity[:, :, 2].reshape(-1),
    }

    maximum_abs_velocity = max(
        4.0 * sigma_nm_ps,
        max(np.max(np.abs(values)) for values in components.values()),
    )

    theory_axis = np.linspace(
        -maximum_abs_velocity,
        maximum_abs_velocity,
        600,
    )
    theory_pdf = theoretical_component_pdf(theory_axis)

    # Generate one separate figure for each component.
    for component_name, values in components.items():
        sample_mean = np.mean(values)
        sample_std = np.std(values, ddof=1)

        plt.figure(figsize=(8, 5))

        plt.hist(
            values,
            bins=N_BINS,
            density=True,
            alpha=0.55,
            edgecolor="black",
            label=(
                f"MD {component_name}: "
                f"mean={sample_mean:.4f}, std={sample_std:.4f}"
            ),
        )

        plt.plot(
            theory_axis,
            theory_pdf,
            linestyle="--",
            linewidth=2.5,
            label=(
                "Theoretical Gaussian: "
                f"mean=0, std={sigma_nm_ps:.4f}"
            ),
        )

        plt.xlabel(f"{component_name} (nm/ps)")
        plt.ylabel("Probability density")
        plt.title(
            f"{component_name} distribution: "
            f"N={n_particles}, T={TEMPERATURE:.0f} K, L={BOX_LENGTH} nm"
        )
        plt.xlim(-maximum_abs_velocity, maximum_abs_velocity)
        plt.ylim(bottom=0)
        plt.grid(linestyle="--", alpha=0.4)
        plt.legend(fontsize=9)
        plt.tight_layout()

        output_file = OUTPUT_DIR / (
            f"{component_name}_N{n_particles}_"
            f"T{int(TEMPERATURE)}_L{BOX_LENGTH}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_file}")

    # Generate an additional overlay figure containing vx, vy, and vz.
    plt.figure(figsize=(9, 6))

    for component_name, values in components.items():
        plt.hist(
            values,
            bins=N_BINS,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=(
                f"MD {component_name}: "
                f"std={np.std(values, ddof=1):.4f}"
            ),
        )

    plt.plot(
        theory_axis,
        theory_pdf,
        linestyle="--",
        linewidth=2.5,
        label=f"Theory: std={sigma_nm_ps:.4f}",
    )

    plt.xlabel("Velocity component (nm/ps)")
    plt.ylabel("Probability density")
    plt.title(
        f"Velocity components: N={n_particles}, "
        f"T={TEMPERATURE:.0f} K, L={BOX_LENGTH} nm"
    )
    plt.xlim(-maximum_abs_velocity, maximum_abs_velocity)
    plt.ylim(bottom=0)
    plt.grid(linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()

    output_file = OUTPUT_DIR / (
        f"components_overlay_N{n_particles}_"
        f"T{int(TEMPERATURE)}_L{BOX_LENGTH}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# Main program
# ============================================================


def main():
    print(
        "Theoretical component standard deviation: "
        f"{sigma_nm_ps:.6f} nm/ps"
    )

    for n_particles in PARTICLE_NUMBERS:
        plot_case(n_particles)

    print(f"\nFinished. Figures are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()