import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'Year': [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Average Development Time': [91, 369, 92, 179, 248, 236, 340, 327, 240, 558, 287, 389, 436, 503, 452, 520, 487, 657, 700, 632, 769, 691, 620, 672, 735, 823, 893, 753, 791, 729, 852, 662, 788, 856, 761, 896, 932, 992, 1001, 1051, 1233, 1353, 1272, 1269, 1455, 1521],
    'Average Team Size': [3, 2, 4, 2, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15, 19, 23, 26, 30, 37, 42, 40, 46, 48, 50, 55, 68, 64, 70, 63, 63, 70, 74, 91, 100, 89, 94, 105, 94, None, None, None, None, None, None, None]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建主坐标轴和次坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制主坐标轴（Average Development Time）为柱状图
ax1.bar(df['Year'], df['Average Development Time'], color='blue', label='Average Development Time')
ax1.set_ylabel('Average Development Time (Days)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 创建次坐标轴
ax2 = ax1.twinx()

# 绘制次坐标轴（Average Team Size）为折线图
ax2.plot(df['Year'], df['Average Team Size'], color='red', linestyle='-', label='Average Team Size')
ax2.set_ylabel('Average Team Size', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 保存图像，dpi=1200
output_file = 'development_time_and_team_size.png'
plt.savefig(output_file, dpi=1200, bbox_inches='tight')

# 显示图表
plt.show()