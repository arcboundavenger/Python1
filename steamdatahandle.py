import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取 Excel 文件
file_path = 'Steam_all_premium_games_detailed_all.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 数据处理
subgenre_agg = df.groupby('Sub-Genre').agg(
    appid_count=('appid', 'count'),
    revenue_mean=('Revenue', 'mean'),
    revenue_sum=('Revenue', 'sum')
).reset_index()

# 准备聚类数据
X = subgenre_agg[['appid_count', 'revenue_mean', 'revenue_sum']]

# 直接应用 K-means 聚类，分成 4 个簇
best_n_clusters = 4
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
subgenre_agg['Cluster'] = kmeans.fit_predict(X)

# 准备绘图数据
x = subgenre_agg['appid_count']  # 横轴：appid 的数量
y = subgenre_agg['revenue_mean']  # 纵轴：revenue 的平均数
sizes = subgenre_agg['revenue_sum'] / 1e6  # 气泡大小：revenue 之和（以百万为单位）

# 绘制气泡图
plt.figure(figsize=(8, 8))  # 设置图形尺寸为正方形
scatter = plt.scatter(x, y, s=sizes, alpha=0.5, c=subgenre_agg['Cluster'], cmap='viridis', edgecolors='w')
plt.xscale('log')  # 设置横轴为对数坐标
plt.yscale('log')  # 设置纵轴为对数坐标
plt.title('')
plt.xlabel('Number of App IDs (Log Scale)')
plt.ylabel('Average Revenue (Log Scale)')

plt.grid(True, linestyle='--', linewidth=0.5)

# 添加图例
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

# 添加标签
for i in range(len(subgenre_agg)):
    plt.annotate(
        subgenre_agg['Sub-Genre'].iloc[i],
        (x.iloc[i], y.iloc[i]),
        fontsize=9,
        ha='right',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
    )

# 设置坐标轴范围
plt.xlim(10**0, 10**4)  # 设置横轴范围
plt.ylim(10**4, 10**8)  # 设置纵轴范围

plt.show()