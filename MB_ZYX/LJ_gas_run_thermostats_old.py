#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one thermostat comparison trajectory for the Lennard-Jones gas.

Set THERMOSTAT to one of:
    "langevin", "andersen", "berendsen", "nve"

The output names match the existing mb_results convention.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R

from LJ_gas_with_thermostats import (
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    simulate_NVT_andersen_step,
    simulate_NVT_berendsen_step,
    initialize_positions,
    initialize_velocities,
    calculate_force,
    density,
    write_xyz_trajectory,
    potential_energy,
    kinetic_energy,
    instantaneous_temperature,
    ideal_gas_pressure,
)


# ============================================================
# Parameters
# ============================================================

N_PARTICLES = 200
MASS_ARGON = 39.95
SIGMA_ARGON = 0.34
EPSILON_ARGON = 120.0 * R * 1e-3

DT_PS = 0.1
N_STEPS = 1000
TEMPERATURE_K = 300.0
BOX_LENGTH_NM = 100.0
TAU_THERMOSTAT_PS = 1.0
RIJ_MIN_NM = 1e-2

# Change only this line to run another method.
THERMOSTAT = "berendsen"

# Using the same seed makes the initial positions and velocities reproducible.
RANDOM_SEED = 20260722

VALID_THERMOSTATS = {
    "langevin",
    "andersen",
    "berendsen",
    "nve",
}

OUTPUT_DIR = Path("mb_results")


# ============================================================
# Main program
# ============================================================


def main() -> None:
    thermostat = THERMOSTAT.lower()
    if thermostat not in VALID_THERMOSTATS:
        raise ValueError(
            f"Unknown thermostat {THERMOSTAT!r}. "
            f"Choose from {sorted(VALID_THERMOSTATS)}."
        )

    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_name_base = OUTPUT_DIR / (
        f"mb_T{TEMPERATURE_K:g}_N{N_PARTICLES}_{thermostat}"
    )

    sim = SimulationParameters(
        dt=DT_PS,
        n_steps=N_STEPS,
        temperature=TEMPERATURE_K,
        box_length=BOX_LENGTH_NM,
        tau_thermostat=TAU_THERMOSTAT_PS,
        rij_min=RIJ_MIN_NM,
    )

    ps = ParticleSystem(N_PARTICLES)
    for particle_index in range(N_PARTICLES):
        ps.set_parameters(
            particle_index,
            mass=MASS_ARGON,
            sigma=SIGMA_ARGON,
            epsilon=EPSILON_ARGON,
        )

    initialize_positions(ps, sim.box_length)
    initialize_velocities(ps, sim.temperature)
    calculate_force(ps, sim)

    rho = density(ps, sim)

    position_trajectory = np.zeros(
        (sim.n_steps + 1, N_PARTICLES, 3),
        dtype=float,
    )
    velocity_trajectory = np.zeros_like(position_trajectory)
    energy_trajectory = np.zeros((sim.n_steps + 1, 4), dtype=float)

    position_trajectory[0] = ps.position
    velocity_trajectory[0] = ps.velocity
    energy_trajectory[0] = (
        potential_energy(ps, sim),
        kinetic_energy(ps),
        instantaneous_temperature(ps),
        ideal_gas_pressure(ps, sim),
    )

    start_time = time.time()

    for step in range(sim.n_steps):
        if thermostat == "langevin":
            simulate_NVT_step(ps, sim)
        elif thermostat == "andersen":
            simulate_NVT_andersen_step(ps, sim)
        elif thermostat == "berendsen":
            simulate_NVT_berendsen_step(ps, sim)
        else:
            simulate_NVE_step(ps, sim)

        position_trajectory[step + 1] = ps.position
        velocity_trajectory[step + 1] = ps.velocity
        energy_trajectory[step + 1] = (
            potential_energy(ps, sim),
            kinetic_energy(ps),
            instantaneous_temperature(ps),
            ideal_gas_pressure(ps, sim),
        )

    elapsed_time = time.time() - start_time

    write_xyz_trajectory(
        str(file_name_base) + "_pos.xyz",
        position_trajectory,
        atom_symbol="Ar",
    )
    np.save(str(file_name_base) + "_vel.npy", velocity_trajectory)
    np.save(str(file_name_base) + "_ene.npy", energy_trajectory)
    np.savetxt(
        str(file_name_base) + "_ene.dat",
        energy_trajectory,
        fmt="%.8e",
        header="# E_pot[kJ/mol] E_kin[kJ/mol] T[K] P[Pa]",
        comments="",
    )

    time_ps = np.arange(sim.n_steps + 1) * sim.dt
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(time_ps, energy_trajectory[:, 2])
    axis.axhline(TEMPERATURE_K, linestyle="--", label="Target T")
    axis.set_xlabel("Time (ps)")
    axis.set_ylabel("Instantaneous temperature (K)")
    axis.set_title(f"{thermostat.capitalize()} temperature trajectory")
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        str(file_name_base) + "_T.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    ensemble = "NVE" if thermostat == "nve" else "NVT"
    output_lines = [
        "",
        "----------------------------------------------------------",
        "Simulation parameters",
        "----------------------------------------------------------",
        f"{'Number of particles:':<30}{ps.n:>10d}",
        f"{'Box length:':<30}{sim.box_length:>10.3f} nm",
        f"{'Density:':<30}{rho:>10.3e} g/cm^3",
        f"{'Time step:':<30}{sim.dt:>10.3f} ps",
        f"{'Number of time steps:':<30}{sim.n_steps:>10d}",
        f"{'Ensemble:':<30}{ensemble:>10}",
        f"{'Method:':<30}{thermostat:>10}",
        f"{'Reference temperature:':<30}{sim.temperature:>10.1f} K",
        f"{'Coupling/collision time:':<30}{sim.tau_thermostat:>10.3f} ps",
        f"{'Random seed:':<30}{RANDOM_SEED:>10d}",
        f"{'Elapsed time:':<30}{elapsed_time:>10.3f} s",
        f"{'Time stamp:':<30}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "----------------------------------------------------------",
        "END",
        "----------------------------------------------------------",
    ]

    for line in output_lines:
        print(line)

    Path(str(file_name_base) + ".out").write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nVelocity trajectory: {file_name_base}_vel.npy")


if __name__ == "__main__":
    main()