#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R

# ==========================================
# 1. 参数配置（必须与当前模拟参数完全一致）
# ==========================================
temperature = 300.0       # 模拟设定温度 K
mass_argon = 39.95        # 氩气摩尔质量 g/mol
file_name_vel = "my_simulation_vel.npy"

# 计算理论分布所需的参数
# 速度单位转换：MD中速度为 nm/ps (即 1000 m/s)，(v in m/s)^2 = 1e6 * (v in nm/ps)^2
M_kg = mass_argon * 1e-3  # 转换为 kg/mol
std_dev_ms = np.sqrt(R * temperature / M_kg)  # SI单位 (m/s) 下的标准差
std_dev_nm_ps = std_dev_ms * 1e-3             # 转换为代码单位 (nm/ps) 下的标准差

# ==========================================
# 2. 读取并清洗数据
# ==========================================
# 读取保存的速度矩阵，形状为 (n_steps + 1, n_particles, 3)
vel_data = np.load(file_name_vel, allow_pickle=True)

# 【关键点】剔除前 30% 的步数作为平衡期 (Equilibration)，只统计平衡后的系统速度
eq_steps = int(0.3 * len(vel_data))
stable_vel = vel_data[eq_steps:]

# 提取并展平所有的速度分量
vx = stable_vel[:, :, 0].flatten()
vy = stable_vel[:, :, 1].flatten()
vz = stable_vel[:, :, 2].flatten()

# 计算所有粒子的速率 (Speed) v = sqrt(vx^2 + vy^2 + vz^2)
speeds = np.sqrt(vx**2 + vy**2 + vz**2)

# ==========================================
# 3. 绘制 3D 速率分布与理论曲线对比
# ==========================================
plt.figure(figsize=(9, 5))

# 绘制 MD 模拟得到的速度直方图 (density=True 保证归一化，面积为1)
plt.hist(speeds, bins=60, density=True, alpha=0.6, color='royalblue', 
         edgecolor='black', label='MD Simulation Data')

# 计算理论 3D 麦克斯韦-玻尔兹曼概率密度
v_range = np.linspace(0, np.max(speeds) * 1.1, 500)
f_v_theoretical = (np.sqrt(2 / np.pi) * (1 / std_dev_nm_ps)**3 * v_range**2 * np.exp(-v_range**2 / (2 * std_dev_nm_ps**2)))

# 绘制理论曲线
plt.plot(v_range, f_v_theoretical, color='crimson', lw=2.5, 
         label=f'Theoretical MB Curve ($T={temperature}$ K)')

plt.title('Maxwell-Boltzmann Speed Distribution (3D)', fontsize=14)
plt.xlabel('Speed $v$ (nm/ps)', fontsize=12)
plt.ylabel('Probability Density $P(v)$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('mb_speed_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 4. 绘制各分量 (vx, vy, vz) 与高斯分布对比
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
components = [vx, vy, vz]
labels = ['$v_x$', '$v_y$', '$v_z$']
colors = ['forestgreen', 'darkorange', 'purple']

# 一维理论高斯分布曲线
v_comp_range = np.linspace(np.min(vx), np.max(vx), 500)
f_v_comp_theoretical = (1 / (std_dev_nm_ps * np.sqrt(2 * np.pi)) * np.exp(-v_comp_range**2 / (2 * std_dev_nm_ps**2)))

for i, ax in enumerate(axes):
    ax.hist(components[i], bins=50, density=True, alpha=0.5, color=colors[i], 
            edgecolor='black', label=f'MD {labels[i]}')
    ax.plot(v_comp_range, f_v_comp_theoretical, color='black', linestyle='--', lw=2, 
            label='1D Theoretical Gaussian')
    ax.set_xlabel(f'Velocity Component {labels[i]} (nm/ps)', fontsize=11)
    if i == 0:
        ax.set_ylabel('Probability Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Component-wise Velocity Distributions vs. 1D Gaussian Theory', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('mb_components_distribution.png', dpi=300, bbox_inches='tight')
plt.show()