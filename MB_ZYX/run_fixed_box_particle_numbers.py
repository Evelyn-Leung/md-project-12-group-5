#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed-box Maxwell-Boltzmann particle-number experiment.

Runs three Langevin NVT simulations with:
    T = 300 K
    L = 100 nm
    N = 20, 50, 100

Place this file in the same directory as LJ_gas.py and run:
    python run_fixed_box_particle_numbers.py
"""

from pathlib import Path

import numpy as np
from scipy.constants import R

from LJ_gas import (
    ParticleSystem,
    SimulationParameters,
    calculate_force,
    initialize_positions,
    initialize_velocities,
    simulate_NVT_step,
)


# ============================================================
# Simulation settings
# ============================================================

TEMPERATURE = 300.0
PARTICLE_NUMBERS = (20, 50, 100)
BOX_LENGTH = 100.0

MASS_ARGON = 39.95
SIGMA_ARGON = 0.34
EPSILON_ARGON = 120.0 * R * 1e-3

DT = 0.1
N_STEPS = 1000
TAU_THERMOSTAT = 1.0
RIJ_MIN = 1e-2

OUTPUT_DIR = Path("mb_results_fixed_L100")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_nvt_simulation(n_particles: int, random_seed: int) -> None:
    """Run one fixed-volume Langevin NVT simulation."""

    output_file = OUTPUT_DIR / (
        f"mb_T{int(TEMPERATURE)}_N{n_particles}_"
        f"L{int(BOX_LENGTH)}_langevin_vel.npy"
    )

    if output_file.exists():
        print(f"Skip existing file: {output_file}")
        return

    np.random.seed(random_seed)

    sim = SimulationParameters(
        dt=DT,
        n_steps=N_STEPS,
        temperature=TEMPERATURE,
        box_length=BOX_LENGTH,
        tau_thermostat=TAU_THERMOSTAT,
        rij_min=RIJ_MIN,
    )

    ps = ParticleSystem(n_particles)

    for particle_index in range(n_particles):
        ps.set_parameters(
            particle_index,
            mass=MASS_ARGON,
            sigma=SIGMA_ARGON,
            epsilon=EPSILON_ARGON,
        )

    initialize_positions(ps, BOX_LENGTH)
    initialize_velocities(ps, TEMPERATURE)
    calculate_force(ps, sim)

    velocity_trajectory = np.zeros(
        (N_STEPS + 1, n_particles, 3),
        dtype=float,
    )
    velocity_trajectory[0] = ps.velocity

    print(
        f"Starting: T={TEMPERATURE:g} K, "
        f"N={n_particles}, L={BOX_LENGTH:g} nm, "
        "ensemble=NVT, thermostat=Langevin"
    )

    for step in range(N_STEPS):
        simulate_NVT_step(ps, sim)
        velocity_trajectory[step + 1] = ps.velocity

        if (step + 1) % 100 == 0:
            print(
                f"  N={n_particles}: "
                f"completed {step + 1}/{N_STEPS} steps"
            )

    np.save(output_file, velocity_trajectory)

    print(
        f"Saved: {output_file} | "
        f"shape={velocity_trajectory.shape}\n"
    )


def main() -> None:
    for index, n_particles in enumerate(PARTICLE_NUMBERS):
        run_nvt_simulation(
            n_particles=n_particles,
            random_seed=20260722 + index,
        )

    print("All simulations are complete.")


if __name__ == "__main__":
    main()
