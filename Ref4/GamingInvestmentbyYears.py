import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'observation_date': ['20Q1', '20Q2', '20Q3', '20Q4', '21Q1', '21Q2', '21Q3', '21Q4', '22Q1', '22Q2', '22Q3', '22Q4', '23Q1', '23Q2', '23Q3', '23Q4', '24Q1', '24Q2', '24Q3', '24Q4'],
    'Private Placement Deals': [88, 78, 101, 98, 148, 133, 131, 167, 186, 140, 115, 120, 149, 99, 89, 78, 127, 132, 120, 87],
    'M&A Deals': [50, 39, 55, 78, 86, 76, 83, 82, 91, 44, 64, 40, 43, 31, 24, 26, 32, 43, 44, 28],
    'Private Placement (Corporate, VC & PE) Value': [0.9, 0.8, 3.6, 1.2, 2.4, 2.4, 4.1, 3.5, 3.5, 4.2, 2.0, 1.2, 1.0, 0.5, 0.8, 0.5, 2.3, 1.0, 1.0, 0.6],
    'M&A (Control & Minority) Value': [2.4, 1.4, 4.5, 4.2, 14.7, 8.1, 6.1, 9.8, 11.1, 86.0, 7.0, 5.0, 0.7, 0.6, 7.1, 0.4, 2.3, 0.7, 0.7, 4.9]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 创建主坐标轴和次坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制堆叠柱状图（主坐标轴）
ax1.bar(df['observation_date'], df['Private Placement (Corporate, VC & PE) Value'], label='Private Placement Value', color='skyblue')
ax1.bar(df['observation_date'], df['M&A (Control & Minority) Value'], bottom=df['Private Placement (Corporate, VC & PE) Value'], label='M&A Value', color='orange')

# 设置主坐标轴标签
ax1.set_ylabel('Value (Billions of USD)')

# 创建次坐标轴
ax2 = ax1.twinx()

# 绘制折线图（次坐标轴）
ax2.plot(df['observation_date'], df['Private Placement Deals'], label='Private Placement Deals', color='blue', marker='o')
ax2.plot(df['observation_date'], df['M&A Deals'], label='M&A Deals', color='red', marker='o')

# 设置次坐标轴标签
ax2.set_ylabel('Number of Deals')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 保存图像为PNG文件
plt.savefig('GamingInvestmentbyYears.png', dpi=1200, bbox_inches='tight')  # 保存为PNG文件，DPI为300

# 显示图表
plt.show()