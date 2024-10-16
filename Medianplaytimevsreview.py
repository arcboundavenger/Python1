import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'Gen9_5000.xlsx'
df = pd.read_excel(file_path, sheet_name='All')

# 将相关列转换为数值型，如果无法转换则设置为 NaN
df['reviewScore'] = pd.to_numeric(df['reviewScore'], errors='coerce')
df['ChineseReviewScore'] = pd.to_numeric(df['ChineseReviewScore'], errors='coerce')
df['EnglishReviewScore'] = pd.to_numeric(df['EnglishReviewScore'], errors='coerce')
df['medianPlaytime'] = pd.to_numeric(df['medianPlaytime'], errors='coerce')

# 过滤掉 medianPlaytime 为 0 或大于 110 小时的数据点
df = df[(df['medianPlaytime'] > 0) & (df['medianPlaytime'] <= 110)]

# 按照 medianPlaytime 每 5 小时分组
df['medianPlaytimeGroup'] = (df['medianPlaytime'] // 5) * 5

# 计算每组的平均值，跳过 NaN
grouped = df.groupby('medianPlaytimeGroup').mean(numeric_only=True)[['reviewScore', 'ChineseReviewScore', 'EnglishReviewScore']]

# 计算每个时间分组的游戏数量
game_counts = df['medianPlaytimeGroup'].value_counts().sort_index()

# 绘制折线图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 主坐标轴：绘制平均评分的折线图
ax1.plot(grouped.index, grouped['reviewScore'], marker='o', label='Review Score', color='b')
ax1.plot(grouped.index, grouped['ChineseReviewScore'], marker='o', label='Chinese Review Score', linestyle='--', color='g')
ax1.plot(grouped.index, grouped['EnglishReviewScore'], marker='o', label='English Review Score', linestyle=':', color='r')

# 设置主坐标轴标题和标签
ax1.set_title('Average Review Scores and Game Counts by Median Playtime Group (Up to 110 Hours)')
ax1.set_xlabel('Median Playtime (hours)')
ax1.set_ylabel('Average Score')
ax1.legend(loc='upper left')
ax1.grid()

# 次坐标轴：绘制游戏数量的柱状图
ax2 = ax1.twinx()  # 创建共享 x 轴的次坐标轴
ax2.bar(game_counts.index, game_counts.values, alpha=0.3, color='gray', label='Game Count', width=4)

# 设置次坐标轴标题和标签
ax2.set_ylabel('Number of Games')
ax2.legend(loc='upper right')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()