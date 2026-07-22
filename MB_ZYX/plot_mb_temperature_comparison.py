#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R

# ==========================================
# 1. 配置对比组实验参数
# ==========================================
mass_argon = 39.95  # 氩气摩尔质量 g/mol
M_kg = mass_argon * 1e-3  # kg/mol

# 将你要对比的：(设定温度, 对应的文件名, 图表展示颜色, 标签名称) 列在字典或列表里
# 注意：核对这里的文件名确实存在于你的左侧文件夹中！
experiments = [
    {"T": 100.0, "file": "my_simulation_100K_vel.npy", "color": "royalblue", "label": "100 K (Cold)"},
    {"T": 300.0, "file": "my_simulation_300K_vel.npy", "color": "forestgreen", "label": "300 K (Room T)"}, # 如果改名了请替换
    {"T": 600.0, "file": "my_simulation_600K_vel.npy", "color": "crimson", "label": "600 K (Hot)"}
]

# ==========================================
# 2. 开始构建同一张大图
# ==========================================
plt.figure(figsize=(10, 6))

# 设置统一的横坐标范围（以 600K 的最大速率再多加 10% 作为上限）
max_v_limit = 0.0

for exp in experiments:
    T = exp["T"]
    file_name = exp["file"]
    color = exp["color"]
    label = exp["label"]
    
    try:
        # 1. 读取并截取平衡态速度数据 (剔除前 30%)
        vel_data = np.load(file_name, allow_pickle=True)
        eq_steps = int(0.3 * len(vel_data))
        stable_vel = vel_data[eq_steps:]
        
        # 2. 计算速率
        vx = stable_vel[:, :, 0].flatten()
        vy = stable_vel[:, :, 1].flatten()
        vz = stable_vel[:, :, 2].flatten()
        speeds = np.sqrt(vx**2 + vy**2 + vz**2)
        
        if np.max(speeds) > max_v_limit:
            max_v_limit = np.max(speeds)
            
        # 3. 绘制 MD 实验直方图 (使用半透明填充与边框，防止颜色覆盖)
        plt.hist(speeds, bins=60, density=True, alpha=0.25, color=color, 
                 histtype='stepfilled', edgecolor=color, linewidth=1.5,
                 label=f'MD Simulation ({label})')
        
        # 4. 计算并绘制该温度下的麦克斯韦-玻尔兹曼理论曲线
        std_dev_ms = np.sqrt(R * T / M_kg)
        std_dev_nm_ps = std_dev_ms * 1e-3
        
        v_range = np.linspace(0, 1.5, 500) # 暂定范围，最后统一裁切
        f_v_theoretical = (np.sqrt(2 / np.pi) * (1 / std_dev_nm_ps)**3 * 
                           v_range**2 * np.exp(-v_range**2 / (2 * std_dev_nm_ps**2)))
        
        # 画出深色理论实线
        plt.plot(v_range, f_v_theoretical, color=color, lw=2.5, linestyle='-',
                 label=f'Theoretical ({label})')
                 
    except FileNotFoundError:
        print(f"Warning: File '{file_name}' not found. Skipping this experiment.")
# ==========================================
# 3. 图表排版与美化
# ==========================================
plt.title('Maxwell-Boltzmann Speed Distributions at Different Temperatures', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Speed $v$ (nm/ps)', fontsize=12)
plt.ylabel('Probability Density $P(v)$', fontsize=12)

# 动态调整横坐标显示范围
if max_v_limit > 0:
    plt.xlim(0, max_v_limit * 1.1)
plt.ylim(bottom=0)

# 优化图例 (分两列显示，更加整洁)
plt.legend(fontsize=10, ncol=2, loc='upper right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)

# 保存高清晰度大图
plt.tight_layout()
plt.savefig('mb_temperature_comparison.png', dpi=300)
plt.show()