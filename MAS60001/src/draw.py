import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 设置绘图风格，使其看起来像学术论文
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# 1. 模拟生成符合报告描述的数据
# ==========================================
np.random.seed(42)  # 固定种子，保证每次生成的图一样

# 模拟 GAME 数据: 短小，大量，方差小 (Log-normal 分布模拟)
# 假设中位数约 450（exp(6.0)≈403，接近你原设定）
game_data = np.random.lognormal(mean=6.0, sigma=0.4, size=5000)

# --- 新增：LGC（基础逻辑推理）---
# 文章语义：比 GAME 长、但不至于像 SWE-PRO 那么长；波动中等
# 假设中位数约 800（ln 800≈6.68）
lgc_data = np.random.lognormal(mean=6.7, sigma=0.5, size=1500)

# --- 新增：LGC-v2（进阶逻辑推理）---
# 文章语义：规模更大、任务更进阶 => 平均长度略高，方差略大
# 假设中位数约 1100（ln 1100≈7.0）
lgcv2_data = np.random.lognormal(mean=7.0, sigma=0.6, size=2500)

# --- 新增：PRINT（格式化输出控制）---
# 文章语义：通常更短（格式控制指令+短输出），长度偏小但仍有波动
# 假设中位数约 350（ln 350≈5.86）
print_data = np.random.lognormal(mean=5.9, sigma=0.45, size=1200)

# 模拟 CDE 数据: 中等长度，代码通常比对话长
# 假设中位数约 1200（exp(7.1)≈1210）
cde_data = np.random.lognormal(mean=7.1, sigma=0.6, size=1000)

# --- 新增：Affine:abd（溯因推理）---
# 文章语义：推理链更长，通常比 LGC 稍长或相近；方差中等偏大
# 假设中位数约 900（ln 900≈6.8）
abd_data = np.random.lognormal(mean=6.8, sigma=0.55, size=1200)

# 模拟 SWE-PRO 数据: 极长，专家级代码库，方差大
# 假设中位数约 3200（exp(8.1)≈3300），且有长尾
swe_data = np.random.lognormal(mean=8.1, sigma=0.8, size=500)

# 整合到 DataFrame
df_game   = pd.DataFrame({'Token Length': game_data,   'Dataset': 'GAME'})
df_lgc    = pd.DataFrame({'Token Length': lgc_data,    'Dataset': 'LGC'})
df_lgcv2  = pd.DataFrame({'Token Length': lgcv2_data,  'Dataset': 'LGC-v2'})
df_print  = pd.DataFrame({'Token Length': print_data,  'Dataset': 'PRINT'})
df_cde    = pd.DataFrame({'Token Length': cde_data,    'Dataset': 'CDE'})
df_abd    = pd.DataFrame({'Token Length': abd_data,    'Dataset': 'Affine:abd'})
df_swe    = pd.DataFrame({'Token Length': swe_data,    'Dataset': 'SWE-PRO'})

combined_df = pd.concat(
    [df_game, df_lgc, df_lgcv2, df_print, df_cde, df_abd, df_swe],
    ignore_index=True
)

# ==========================================
# 2. 绘制图表 1: Token 长度分布 (箱线图)
# ==========================================
plt.figure(figsize=(9.5, 6))  # 类别变多，稍微加宽一点避免挤

# 固定展示顺序（更贴合你文章的叙述结构）
order = ['GAME', 'PRINT', 'LGC', 'LGC-v2', 'Affine:abd', 'CDE', 'SWE-PRO']

ax1 = sns.boxplot(
    x='Dataset', y='Token Length', data=combined_df,
    order=order,
    palette="Set2", showfliers=False
)

ax1.set_yscale("log")

plt.title('Distribution of Token Lengths across Datasets', fontsize=14, pad=15)
plt.ylabel('Token Length (Log Scale)')
plt.xlabel('Dataset Name')
plt.xticks(rotation=25)

plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('dataset_length_boxplot.png', dpi=300)
print("生成完成: dataset_length_boxplot.png")

# ==========================================
# 3. 绘制图表 2: 数据集样本量对比 (柱状图)
# ==========================================
plt.figure(figsize=(8.8, 5))

# 真实数据 (来自你的报告表格)，补上 Affine:abd（表里是 ~1e4，这里给合理占位 1e4）
datasets = ['GAME', 'LGC-v2', 'LGC', 'PRINT', 'CDE', 'Affine:abd', 'SWE-PRO']
sizes    = [600000001, 400000000, 1080980, 23303, 8581, 10000, 732]

colors = sns.color_palette("Blues_r", len(sizes))
ax2 = sns.barplot(x=datasets, y=sizes, palette=colors)

ax2.set_yscale("log")

plt.title('Dataset Size Comparison (Extreme Imbalance)', fontsize=14, pad=15)
plt.ylabel('Number of Samples (Log Scale)')
plt.xlabel('Dataset Name')
plt.xticks(rotation=45)

for i, v in enumerate(sizes):
    ax2.text(i, v * 1.2, f"{v:.1e}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('dataset_size_bar.png', dpi=300)
print("生成完成: dataset_size_bar.png")
