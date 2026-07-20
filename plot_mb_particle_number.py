#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R

# ==========================================
# P a r a m e t e r
# ==========================================
temperature = 300.0
mass_argon = 39.95
M_kg = mass_argon * 1e-3

std_dev_ms = np.sqrt(R * temperature / M_kg)
std_dev_nm_ps = std_dev_ms * 1e-3

# Define the 4 experimental datasets
experiments = [
    {"N": 20,  "file": "my_simulation_N20_vel.npy",  "color": "crimson",     "label": "N = 20 (Extreme Small)"},
    {"N": 100, "file": "my_simulation_N100_vel.npy", "color": "darkorange",  "label": "N = 100 (Small)"},
    {"N": 200, "file": "my_simulation_vel.npy", "color": "royalblue",   "label": "N = 200 (Baseline)"},
    {"N": 500, "file": "my_simulation_N500_vel.npy", "color": "forestgreen", "label": "N = 500 (Large)"}
]

# ==========================================
# Build 2x2 grid and calculate RMSE
# ==========================================
# layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes_flat = axes.flatten() # 展平为一维索引方便循环

print("\n" + "="*55)
print("Quantitative Deviation Analysis across System Sizes (MD vs Theory)")
print("="*55)

for i, exp in enumerate(experiments):
    N = exp["N"]
    file_name = exp["file"]
    color = exp["color"]
    label = exp["label"]
    ax = axes_flat[i]
    
    try:
        # 1. Load data and discard the first 30% for equilibration
        vel_data = np.load(file_name, allow_pickle=True)
        eq_steps = int(0.3 * len(vel_data))
        stable_vel = vel_data[eq_steps:]
        
        # 2. Calculate speeds v = sqrt(vx^2 + vy^2 + vz^2)
        vx = stable_vel[:, :, 0].flatten()
        vy = stable_vel[:, :, 1].flatten()
        vz = stable_vel[:, :, 2].flatten()
        speeds = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # 3. Plot histogram
        counts, bin_edges, _ = ax.hist(speeds, bins=45, density=True, alpha=0.45, 
                                       color=color, edgecolor='black', label='MD Data')
        
        # 4. Calculate theoretical error (RMSE)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        f_v_bin_theory = (np.sqrt(2 / np.pi) * (1 / std_dev_nm_ps)**3 * 
                          bin_centers**2 * np.exp(-bin_centers**2 / (2 * std_dev_nm_ps**2)))
        rmse = np.sqrt(np.mean((counts - f_v_bin_theory)**2))
        
        print(f"👉 System Size: {label:<25} | Theoretical Deviation (RMSE): {rmse:.5f}")
        
        # 5. Plot theoretical mb curve
        v_range = np.linspace(0, np.max(speeds)*1.1, 300)
        f_v_continuous = (np.sqrt(2 / np.pi) * (1 / std_dev_nm_ps)**3 * 
                          v_range**2 * np.exp(-v_range**2 / (2 * std_dev_nm_ps**2)))
        ax.plot(v_range, f_v_continuous, color='black', lw=2.2, linestyle='--', label='Theory (300K)')
        
        # 6. Annotate RMSE
        ax.text(0.50, 0.82, f"RMSE: {rmse:.4f}\n($1/\\sqrt{{N}} = {1/np.sqrt(N):.3f}$)", 
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
                fontsize=10, fontweight='bold')
                
    except FileNotFoundError:
        print(f"File not found: {file_name}. Please check if the simulation for N={N} has been executed!")
        
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Set labels
    if i in [2, 3]:
        ax.set_xlabel('Speed $v$ (nm/ps)', fontsize=11)
    if i in [0, 2]:
        ax.set_ylabel('Probability Density $P(v)$', fontsize=11)

print("="*55 + "\n")
plt.suptitle('Convergence of Maxwell-Boltzmann Distribution across System Sizes ($N=20 \\to 500$)', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('mb_particle_number_4grids.png', dpi=300)
plt.show()