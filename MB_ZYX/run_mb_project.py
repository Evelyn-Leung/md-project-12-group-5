#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the minimum set of MD simulations for a Maxwell-Boltzmann project.

Place this file in the same folder as LJ_gas.py and run:
    python run_mb_project.py

It creates velocity files named, for example:
    mb_T300_N200_langevin_vel.npy
"""

from pathlib import Path
import numpy as np
from scipy.constants import R

from LJ_gas import (
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    initialize_positions,
    initialize_velocities,
    calculate_force,
)

# -------------------------
# Common simulation settings
# -------------------------
MASS_ARGON = 39.95          # g/mol
SIGMA_ARGON = 0.34          # nm
EPSILON_ARGON = 120 * R * 1e-3  # kJ/mol

DT = 0.1                    # ps; same as the supplied program
N_STEPS = 1000              # use 3000-5000 for a final higher-statistics run
TAU_LANGEVIN = 1.0          # ps
ANDERSEN_RATE = 1.0         # collisions per ps
RIJ_MIN = 1e-2              # nm

REFERENCE_N = 200
REFERENCE_BOX = 100.0        # nm
OUTPUT_DIR = Path("mb_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def box_length_at_constant_density(n_particles: int) -> float:
    """Scale L as N^(1/3), so changing N does not also change density."""
    return REFERENCE_BOX * (n_particles / REFERENCE_N) ** (1.0 / 3.0)


def andersen_step(ps, sim, collision_rate=ANDERSEN_RATE):
    """One NVE step followed by an Andersen thermostat collision step."""
    simulate_NVE_step(ps, sim)

    # Exact per-step collision probability for a Poisson process.
    p_collision = 1.0 - np.exp(-collision_rate * sim.dt)
    mask = np.random.random(ps.n) < p_collision
    n_collisions = int(np.sum(mask))

    if n_collisions > 0:
        # sigma = sqrt(RT/M), then convert m/s -> nm/ps by multiplying by 1e-3.
        molar_mass_kg = ps.mass[mask] * 1e-3
        sigma_nm_ps = np.sqrt(R * sim.temperature / molar_mass_kg) * 1e-3
        ps.velocity[mask] = np.random.normal(
            loc=0.0,
            scale=sigma_nm_ps[:, None],
            size=(n_collisions, 3),
        )


def run_case(temperature: float, n_particles: int, thermostat: str, seed: int):
    thermostat = thermostat.lower()
    if thermostat not in {"nve", "langevin", "andersen"}:
        raise ValueError(f"Unknown thermostat: {thermostat}")

    output_file = OUTPUT_DIR / (
        f"mb_T{int(temperature)}_N{n_particles}_{thermostat}_vel.npy"
    )

    # Avoid repeating an expensive run accidentally.
    if output_file.exists():
        print(f"Skip existing file: {output_file}")
        return

    np.random.seed(seed)
    box_length = box_length_at_constant_density(n_particles)

    tau = TAU_LANGEVIN if thermostat == "langevin" else None
    sim = SimulationParameters(
        dt=DT,
        n_steps=N_STEPS,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau,
        rij_min=RIJ_MIN,
    )

    ps = ParticleSystem(n_particles)
    for i in range(n_particles):
        ps.set_parameters(
            i,
            mass=MASS_ARGON,
            sigma=SIGMA_ARGON,
            epsilon=EPSILON_ARGON,
        )

    initialize_positions(ps, sim.box_length)
    initialize_velocities(ps, sim.temperature)
    calculate_force(ps, sim)

    velocity_trajectory = np.zeros((N_STEPS + 1, n_particles, 3), dtype=float)
    velocity_trajectory[0] = ps.velocity

    for step in range(N_STEPS):
        if thermostat == "nve":
            simulate_NVE_step(ps, sim)
        elif thermostat == "langevin":
            simulate_NVT_step(ps, sim)
        else:
            andersen_step(ps, sim)

        velocity_trajectory[step + 1] = ps.velocity

    np.save(output_file, velocity_trajectory)
    print(
        f"Saved {output_file} | T={temperature:g} K, N={n_particles}, "
        f"thermostat={thermostat}, L={box_length:.3f} nm"
    )


def build_experiment_list():
    """Minimum experiment matrix covering every project requirement."""
    cases = []

    # 1. Temperature dependence: fixed N and thermostat.
    for temperature in (100.0, 300.0, 600.0):
        cases.append((temperature, 200, "langevin"))

    # 2. Particle-number dependence: fixed T and thermostat.
    for n_particles in (10, 20, 50, 100, 200):
        cases.append((300.0, n_particles, "langevin"))

    # 3. Thermostat dependence: same T and N.
    for thermostat in ("nve", "langevin", "andersen"):
        cases.append((300.0, 200, thermostat))

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(cases))


if __name__ == "__main__":
    cases = build_experiment_list()
    print(f"Running {len(cases)} unique simulations")

    for index, (temperature, n_particles, thermostat) in enumerate(cases):
        run_case(
            temperature=temperature,
            n_particles=n_particles,
            thermostat=thermostat,
            seed=20260722 + index,
        )