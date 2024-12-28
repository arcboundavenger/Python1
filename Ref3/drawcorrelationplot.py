import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games Duplicate')

# 处理无穷大和缺失值
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# 选择所有变量，去掉指定的列
variables_to_include = data.drop(columns=['AppID', 'Estimated owners', 'Release date'])

# 计算相关系数矩阵
correlation_matrix = variables_to_include.corr()

# 绘制相关系数热图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Heatmap (Excluding AppID, Estimated owners, Release date)')
plt.show()