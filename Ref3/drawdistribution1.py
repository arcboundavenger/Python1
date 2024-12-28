import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games Duplicate')

# 检查是否存在缺失值
print("AveragePlaytime 缺失值统计：", data['AveragePlaytime'].isnull().sum())

# 删除缺失值
data.dropna(subset=['AveragePlaytime'], inplace=True)

# 绘制 AveragePlaytime 的频率分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(data['AveragePlaytime'], bins=50, kde=True, color='blue')  # 使用 seaborn 的 histplot
plt.title('Frequency Distribution of AveragePlaytime')
plt.xlabel('AveragePlaytime')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()