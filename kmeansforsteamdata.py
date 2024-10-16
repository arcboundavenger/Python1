import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 读取Excel表格数据
df = pd.read_excel('Steam_all_premium_games_detailed_edited.xlsx')

# 2. 数据预处理（检查缺失值等，这里假设数据已经是干净的）
# 如果需要，可以添加如下代码来处理缺失值：
# df.fillna(df.mean(), inplace=True)  # 对于数值型列，使用平均值填充缺失值

# 3. K-Means聚类
# 假设我们想要聚成3类
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=None)
df['Cluster'] = kmeans.fit_predict(df[['Count of appid', 'Average Revenue', 'Sum of Revenue']])

# 4. 可视化聚类结果为气泡图
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x='Count of appid', y='Average Revenue', size='Sum of Revenue', legend=False, hue='Cluster', sizes=(50, 5000), palette='tab10')

# 添加数据标签
for line in range(0, df.shape[0]):
    plt.text(df['Count of appid'].iloc[line], df['Average Revenue'].iloc[line], df['Sub-Genre'].iloc[line], horizontalalignment='center', size='medium', color='black')

# 设置对数坐标轴
plt.xscale('log')
plt.yscale('log')

# 设置x轴和y轴的上下限
plt.xlim(1, 10000)
plt.ylim(10000, 100000000)

# 设置x轴和y轴的主要网格线位置
xticks = np.logspace(0, 4, 5)  # 对数尺度下的x轴刻度数组
yticks = np.logspace(4, 8, 5)  # 对数尺度下的y轴刻度数组
plt.xticks(xticks)
plt.yticks(yticks)

# 添加主要网格线
plt.grid(which='major', linestyle=':', linewidth='0.5', color='gray')

plt.show()