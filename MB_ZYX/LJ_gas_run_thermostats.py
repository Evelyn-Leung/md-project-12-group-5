#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one of four MD methods:

    langevin
    andersen
    berendsen
    nve

Output:
    mb_results/mb_T300_N200_<method>_vel.npy
    mb_results/mb_T300_N200_<method>_temperature.npy
    mb_results/mb_T300_N200_<method>_temperature.dat
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
    instantaneous_temperature,
)


# ============================================================
# Settings
# ============================================================

THERMOSTAT = "berendsen"

VALID_THERMOSTATS = {
    "langevin",
    "andersen",
    "berendsen",
    "nve",
}

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

RANDOM_SEED = 20260722

OUTPUT_DIRECTORY = Path("mb_results")


# ============================================================
# Thermostats
# ============================================================

def andersen_thermostat_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
) -> None:
    """
    Andersen thermostat.

    Each particle collides with the heat bath with probability

        p = 1 - exp(-dt/tau).

    A collided particle receives a newly sampled
    Maxwell-Boltzmann velocity.
    """

    if (
        sim.tau_thermostat is None
        or sim.tau_thermostat <= 0.0
    ):
        raise ValueError(
            "Andersen thermostat requires tau_thermostat > 0."
        )

    collision_probability = (
        1.0
        - np.exp(
            -sim.dt
            / sim.tau_thermostat
        )
    )

    collided = (
        np.random.random(ps.n)
        < collision_probability
    )

    number_collided = int(
        np.count_nonzero(collided)
    )

    if number_collided == 0:
        return

    # Convert g/mol to kg/mol.
    mass_kg_mol = (
        ps.mass[collided]
        * 1e-3
    )

    # Maxwell-Boltzmann component standard deviation in m/s.
    stddev_m_s = np.sqrt(
        R
        * sim.temperature
        / mass_kg_mol
    )

    # Convert m/s to nm/ps.
    stddev_nm_ps = (
        stddev_m_s
        * 1e-3
    )

    ps.velocity[collided, :] = np.random.normal(
        loc=0.0,
        scale=stddev_nm_ps[:, np.newaxis],
        size=(number_collided, 3),
    )


def berendsen_thermostat_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
) -> None:
    """
    Berendsen thermostat.

        v_new = lambda * v_old

        lambda^2 =
            1
            + dt/tau
            * (T_target/T_current - 1)
    """

    if (
        sim.tau_thermostat is None
        or sim.tau_thermostat <= 0.0
    ):
        raise ValueError(
            "Berendsen thermostat requires tau_thermostat > 0."
        )

    current_temperature = (
        instantaneous_temperature(ps)
    )

    if current_temperature <= 0.0:
        raise ValueError(
            "Instantaneous temperature must be positive."
        )

    lambda_squared = (
        1.0
        + sim.dt
        / sim.tau_thermostat
        * (
            sim.temperature
            / current_temperature
            - 1.0
        )
    )

    if lambda_squared <= 0.0:
        raise ValueError(
            "Berendsen scaling factor became non-positive. "
            "Reduce dt or increase tau."
        )

    ps.velocity *= np.sqrt(
        lambda_squared
    )


def simulate_andersen_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
) -> None:
    """
    NVE velocity-Verlet propagation followed by
    Andersen heat-bath collisions.
    """

    simulate_NVE_step(ps, sim)
    andersen_thermostat_step(ps, sim)


def simulate_berendsen_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
) -> None:
    """
    NVE velocity-Verlet propagation followed by
    Berendsen velocity rescaling.
    """

    simulate_NVE_step(ps, sim)
    berendsen_thermostat_step(ps, sim)


def propagate_one_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
    method: str,
) -> None:

    if method == "langevin":
        simulate_NVT_step(ps, sim)

    elif method == "andersen":
        simulate_andersen_step(ps, sim)

    elif method == "berendsen":
        simulate_berendsen_step(ps, sim)

    elif method == "nve":
        simulate_NVE_step(ps, sim)

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )


# ============================================================
# Main simulation
# ============================================================

def main() -> None:

    if THERMOSTAT not in VALID_THERMOSTATS:
        raise ValueError(
            f"THERMOSTAT must be one of "
            f"{sorted(VALID_THERMOSTATS)}"
        )

    np.random.seed(
        RANDOM_SEED
    )

    sim = SimulationParameters(
        dt=DT_PS,
        n_steps=N_STEPS,
        temperature=TEMPERATURE_K,
        box_length=BOX_LENGTH_NM,
        tau_thermostat=TAU_THERMOSTAT_PS,
        rij_min=RIJ_MIN_NM,
    )

    ps = ParticleSystem(
        N_PARTICLES
    )

    for particle_index in range(
        N_PARTICLES
    ):
        ps.set_parameters(
            particle_index,
            mass=MASS_ARGON,
            sigma=SIGMA_ARGON,
            epsilon=EPSILON_ARGON,
        )

    initialize_positions(
        ps,
        sim.box_length,
    )

    initialize_velocities(
        ps,
        sim.temperature,
    )

    calculate_force(
        ps,
        sim,
    )

    velocity_trajectory = np.zeros(
        (
            N_STEPS + 1,
            N_PARTICLES,
            3,
        ),
        dtype=float,
    )

    temperature_trajectory = np.zeros(
        N_STEPS + 1,
        dtype=float,
    )

    velocity_trajectory[0] = (
        ps.velocity
    )

    temperature_trajectory[0] = (
        instantaneous_temperature(ps)
    )

    print(
        f"Running {THERMOSTAT}: "
        f"N={N_PARTICLES}, "
        f"T={TEMPERATURE_K:g} K, "
        f"L={BOX_LENGTH_NM:g} nm"
    )

    for step in range(
        N_STEPS
    ):
        propagate_one_step(
            ps,
            sim,
            THERMOSTAT,
        )

        velocity_trajectory[
            step + 1
        ] = ps.velocity

        temperature_trajectory[
            step + 1
        ] = instantaneous_temperature(
            ps
        )

        if (
            (step + 1) % 100 == 0
            or step == 0
        ):
            print(
                f"step {step + 1:4d}/{N_STEPS}: "
                f"T = "
                f"{temperature_trajectory[step + 1]:.3f} K"
            )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_base = (
        OUTPUT_DIRECTORY
        / (
            f"mb_T{TEMPERATURE_K:g}"
            f"_N{N_PARTICLES}"
            f"_{THERMOSTAT}"
        )
    )

    np.save(
        str(file_base) + "_vel.npy",
        velocity_trajectory,
    )

    np.save(
        str(file_base) + "_temperature.npy",
        temperature_trajectory,
    )

    np.savetxt(
        str(file_base) + "_temperature.dat",
        temperature_trajectory,
        fmt="%.10f",
        header="temperature_K",
    )

    equilibrated_temperature = (
        temperature_trajectory[
            int(
                0.30
                * temperature_trajectory.size
            ):
        ]
    )

    print()
    print("Simulation complete")
    print(
        "Mean equilibrated temperature: "
        f"{np.mean(equilibrated_temperature):.6f} K"
    )
    print(
        "Temperature standard deviation: "
        f"{np.std(equilibrated_temperature, ddof=1):.6f} K"
    )
    print(
        "Velocity file:",
        str(file_base) + "_vel.npy",
    )


if __name__ == "__main__":
    main()
