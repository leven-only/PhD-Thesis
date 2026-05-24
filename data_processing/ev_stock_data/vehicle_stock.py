import matplotlib.pyplot as plt
import pandas as pd

# 1. 整理数据
data = {
    'year': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024, 2030, 2030],
    'powertrain': ['BEV', 'PHEV', 'BEV', 'PHEV', 'BEV', 'PHEV', 'BEV', 'PHEV', 'BEV', 'PHEV', 'BEV', 'PHEV'],
    'value': [
        6800000, 3400000,  # 2020
        11000000, 5300000,  # 2021
        18000000, 8000000,  # 2022
        28000000, 12000000,  # 2023
        39000000, 19000000,  # 2024
        150000000, 82000000  # 2030 (Projection)
    ]
}

df = pd.DataFrame(data)
plot_df = df.pivot(index='year', columns='powertrain', values='value').fillna(0)

# 颜色定义
color_bev = '#27ae60'  # 深一点的绿色方便看清文字
color_phev = '#2980b9'  # 深一点的蓝色方便看清文字
bg_bev = '#2ecc71'
bg_phev = '#3498db'

# 3. 开始绘图
plt.figure(figsize=(16, 10))

# 绘制堆叠区域图
plt.stackplot(plot_df.index,
              plot_df['BEV'],
              plot_df['PHEV'],
              labels=['BEV (Battery Electric)', 'PHEV (Plug-in Hybrid)'],
              colors=[bg_bev, bg_phev],
              alpha=0.7)

# 4. 字体大幅加大
font_title = 30
font_label = 22
font_tick = 20
font_text = 20

plt.title('Global EV Stock Projection (2020 - 2030)', fontsize=font_title, pad=35, fontweight='bold')
plt.xlabel('Year', fontsize=font_label, labelpad=15)
plt.ylabel('Number of Vehicles (Millions)', fontsize=font_label, labelpad=15)
plt.legend(loc='upper left', fontsize=font_tick, frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# 设置轴刻度
plt.xticks([2020, 2021, 2022, 2023, 2024, 2030], fontsize=font_tick)


def format_millions(x, pos):
    return f'{int(x / 1e6)}M'


plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
plt.yticks(fontsize=font_tick)

# 5. 在 2024 年和 2030 年标注具体数值和总计
target_years = [2024, 2030]

for year in target_years:
    bev_val = plot_df.loc[year, 'BEV']
    phev_val = plot_df.loc[year, 'PHEV']
    total_val = bev_val + phev_val

    # 标注 BEV 数值 (使用对应绿色)
    # 稍微向上偏一点避免重叠
    plt.text(year, bev_val / 2, f'BEV: {bev_val / 1e6:.0f}M',
             ha='center', va='center', fontsize=font_text, color=color_bev, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    # 标注 PHEV 数值 (使用对应蓝色)
    plt.text(year, bev_val + phev_val / 2, f'PHEV: {phev_val / 1e6:.0f}M',
             ha='center', va='center', fontsize=font_text, color=color_phev, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    # 标注总计 (黑色加粗)
    plt.text(year, total_val + 3000000, f'Total: {total_val / 1e6:.0f}M',
             ha='center', va='bottom', fontsize=font_label, color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('ev_projection_final.png')
plt.close()