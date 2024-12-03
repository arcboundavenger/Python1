import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_name = 'Steam_all_premium_games_detailed_all.xlsx'
data = pd.read_excel(file_name, sheet_name='Sheet1')

# 提取 Revenue 列
revenue = data['Revenue']

# 绘制频率分布直方图
plt.figure(figsize=(10, 6))
plt.hist(revenue, bins=500, color='blue', alpha=0.7)  # 增加柱数以减小宽度
plt.title('Frequency Distribution of Revenue')
plt.xlabel('Revenue (Unit: $)')
plt.ylabel('Number of Games (Log Scale)')
plt.yscale('log')  # 设置纵轴为对数坐标
plt.grid(axis='y', alpha=0.75)
plt.show()