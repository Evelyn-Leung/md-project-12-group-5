#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze velocity trajectories for the Maxwell-Boltzmann MD project.

Run after run_mb_project.py:
    python analyze_mb_project.py

Outputs:
    mb_temperature_comparison.png
    mb_thermostat_comparison.png
    mb_components_300K.png
    mb_small_system_deviation.png
    mb_summary.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R
from scipy.stats import kstest, maxwell, norm

MASS_ARGON = 39.95          # g/mol
M_KG = MASS_ARGON * 1e-3    # kg/mol
RESULT_DIR = Path("mb_results")
BURN_IN_FRACTION = 0.30
FRAME_STRIDE = 5             # reduces time correlation and analysis cost


def sigma_theory_nm_ps(temperature):
    """Theoretical standard deviation of vx, vy, or vz in nm/ps."""
    return np.sqrt(R * temperature / M_KG) * 1e-3


def mb_speed_pdf(v, temperature):
    sigma = sigma_theory_nm_ps(temperature)
    return maxwell.pdf(v, loc=0.0, scale=sigma)


def load_velocity(temperature, n_particles, thermostat):
    file_path = RESULT_DIR / (
        f"mb_T{int(temperature)}_N{n_particles}_{thermostat}_vel.npy"
    )
    velocity = np.load(file_path)
    start = int(BURN_IN_FRACTION * len(velocity))
    return velocity[start::FRAME_STRIDE]


def speed_from_velocity(velocity):
    return np.linalg.norm(velocity, axis=-1)


def measured_temperature(velocity):
    """Temperature from mean kinetic energy using 3N degrees of freedom."""
    mean_v2_nmps = np.mean(np.sum(velocity**2, axis=-1))
    return M_KG * 1e6 * mean_v2_nmps / (3.0 * R)


def framewise_ks_statistics(velocity, temperature):
    """Return per-frame KS deviations for speed and velocity components.

    Per-frame values are used so that a small-N system is not made to look
    artificially perfect merely by pooling thousands of correlated frames.
    """
    sigma = sigma_theory_nm_ps(temperature)
    speed_ks = []
    component_ks = []

    for frame in velocity:
        speeds = np.linalg.norm(frame, axis=1)
        speed_ks.append(
            kstest(speeds, maxwell.cdf, args=(0.0, sigma)).statistic
        )

        frame_component_ks = []
        for axis in range(3):
            frame_component_ks.append(
                kstest(frame[:, axis], norm.cdf, args=(0.0, sigma)).statistic
            )
        component_ks.append(np.mean(frame_component_ks))

    return np.asarray(speed_ks), np.asarray(component_ks)


def plot_speed_group(cases, title, output_name):
    plt.figure(figsize=(9, 5.5))

    all_speeds = []
    loaded = []
    for temperature, n_particles, thermostat, label in cases:
        velocity = load_velocity(temperature, n_particles, thermostat)
        speeds = speed_from_velocity(velocity).ravel()
        all_speeds.append(speeds)
        loaded.append((temperature, n_particles, thermostat, label, speeds))

    max_speed = max(np.percentile(s, 99.9) for s in all_speeds)
    bins = np.linspace(0.0, max_speed, 55)
    v_grid = np.linspace(0.0, max_speed, 500)

    for temperature, n_particles, thermostat, label, speeds in loaded:
        line = plt.hist(
            speeds,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"MD: {label}",
        )
        color = line[2][0].get_edgecolor()
        plt.plot(
            v_grid,
            mb_speed_pdf(v_grid, temperature),
            linestyle="--",
            linewidth=2.0,
            color=color,
            label=f"Theory: {label}",
        )

    plt.xlabel("Speed v (nm/ps)")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()


def plot_components():
    temperature = 300.0
    velocity = load_velocity(temperature, 200, "langevin")
    components = [velocity[:, :, i].ravel() for i in range(3)]
    labels = [r"$v_x$", r"$v_y$", r"$v_z$"]
    sigma = sigma_theory_nm_ps(temperature)

    limit = max(np.percentile(np.abs(c), 99.8) for c in components)
    x = np.linspace(-limit, limit, 500)
    theory = norm.pdf(x, loc=0.0, scale=sigma)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, data, label in zip(axes, components, labels):
        axis.hist(data, bins=55, density=True, alpha=0.55, edgecolor="black")
        axis.plot(x, theory, linestyle="--", linewidth=2, color="black")
        axis.set_xlabel(f"{label} (nm/ps)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Probability density")
    fig.suptitle("Component-wise velocity distributions, 300 K, N=200, Langevin")
    fig.tight_layout()
    fig.savefig("mb_components_300K.png", dpi=300)
    plt.close(fig)


def analyze_all_cases():
    cases = []

    for temperature in (100.0, 300.0, 600.0):
        cases.append((temperature, 200, "langevin"))
    for n_particles in (10, 20, 50, 100, 200):
        cases.append((300.0, n_particles, "langevin"))
    for thermostat in ("nve", "langevin", "andersen"):
        cases.append((300.0, 200, thermostat))
    cases = list(dict.fromkeys(cases))

    rows = []
    for temperature, n_particles, thermostat in cases:
        velocity = load_velocity(temperature, n_particles, thermostat)
        speed_ks, component_ks = framewise_ks_statistics(velocity, temperature)
        t_measured = measured_temperature(velocity)

        rows.append(
            {
                "temperature_target_K": temperature,
                "n_particles": n_particles,
                "thermostat": thermostat,
                "temperature_measured_K": t_measured,
                "temperature_relative_error": abs(t_measured - temperature) / temperature,
                "speed_KS_mean": np.mean(speed_ks),
                "speed_KS_std": np.std(speed_ks, ddof=1),
                "component_KS_mean": np.mean(component_ks),
                "component_KS_std": np.std(component_ks, ddof=1),
            }
        )

    with open("mb_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_small_system_deviation(rows):
    selected = sorted(
        [
            row
            for row in rows
            if row["temperature_target_K"] == 300.0
            and row["thermostat"] == "langevin"
        ],
        key=lambda row: row["n_particles"],
    )

    n_values = np.array([row["n_particles"] for row in selected], dtype=float)
    ks_mean = np.array([row["speed_KS_mean"] for row in selected])
    ks_std = np.array([row["speed_KS_std"] for row in selected])

    # Empirical log-log slope; ideal finite-sampling behavior is approximately N^(-1/2).
    slope, intercept = np.polyfit(np.log(n_values), np.log(ks_mean), 1)
    fitted = np.exp(intercept) * n_values**slope

    plt.figure(figsize=(7, 5))
    plt.errorbar(n_values, ks_mean, yerr=ks_std, marker="o", capsize=4,
                 label="Mean per-frame KS deviation")
    plt.plot(n_values, fitted, linestyle="--",
             label=f"Fit: KS proportional to N^{slope:.2f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of particles N")
    plt.ylabel("KS deviation from Maxwell theory")
    plt.title("Finite-size deviation from Maxwell-Boltzmann theory")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mb_small_system_deviation.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_speed_group(
        [
            (100.0, 200, "langevin", "100 K"),
            (300.0, 200, "langevin", "300 K"),
            (600.0, 200, "langevin", "600 K"),
        ],
        "Maxwell-Boltzmann speed distribution at different temperatures",
        "mb_temperature_comparison.png",
    )

    plot_speed_group(
        [
            (300.0, 200, "nve", "NVE"),
            (300.0, 200, "langevin", "Langevin"),
            (300.0, 200, "andersen", "Andersen"),
        ],
        "Influence of thermostat on the speed distribution",
        "mb_thermostat_comparison.png",
    )

    plot_components()
    summary_rows = analyze_all_cases()
    plot_small_system_deviation(summary_rows)
    print("Analysis complete. See PNG figures and mb_summary.csv")