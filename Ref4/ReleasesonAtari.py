import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'Year': [1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990],
    'Home Game Revenue': [110, 180, 260, 350, 500, 1000, 3200, 1950, 800, 100, 330, 980, 1900, 2600],
    'New Game Releases on Atari 2600': [10, 19, 3, 20, 18, 114, 191, 37, 2, 3, 8, 9, 12, 8]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建主坐标轴和次坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制主坐标轴（Home Game Revenue）
ax1.plot(df['Year'], df['Home Game Revenue'], color='blue', marker='o', linestyle='-', label='Home Game Revenue')
ax1.set_xlabel('Year')
ax1.set_ylabel('Revenue (USD)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 创建次坐标轴
ax2 = ax1.twinx()

# 绘制次坐标轴（New Game Releases on Atari 2600）
ax2.plot(df['Year'], df['New Game Releases on Atari 2600'], color='red', marker='s', linestyle='-', label='New Game Releases on Atari 2600')
ax2.set_ylabel('Number of Releases', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 保存图像，dpi=1200
output_file = 'revenue_and_releases_line_chart.png'
plt.savefig(output_file, dpi=1200, bbox_inches='tight')

# 显示图表
plt.show()