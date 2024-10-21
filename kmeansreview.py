# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 设置字体以支持中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

# 读取 Excel 文件
file_path = 'Gen9_5000.xlsx'
df = pd.read_excel(file_path, sheet_name='All')

# 将相关列转换为数值型，如果无法转换则设置为 NaN
df['ScoreGap'] = pd.to_numeric(df['ScoreGap'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['reviewScore'] = pd.to_numeric(df['reviewScore'], errors='coerce')

# 过滤掉包含 NaN 的行
df = df.dropna(subset=['ScoreGap', 'price', 'reviewScore'])

# 选择要聚类的特征
X = df[['ScoreGap', 'price', 'reviewScore']]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means 聚类（设置聚类数为4）
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 可视化聚类结果（横轴为 price，纵轴为 ScoreGap）
plt.figure(figsize=(12, 8))
plt.scatter(df['price'], df['ScoreGap'], c=df['cluster'], cmap='viridis', alpha=0.6)

# 标注每个点对应的 name
for i in range(len(df)):
    plt.annotate(df['name'].iloc[i], (df['price'].iloc[i], df['ScoreGap'].iloc[i]), fontsize=8, alpha=0.7)

plt.title('K-means Clustering Result (Score Gap vs Price)')
plt.xlabel('Price')
plt.ylabel('Score Gap')
plt.grid()
plt.tight_layout()
plt.show()

# 输出到 Excel
output_file = 'clustered_results_with_steamId.xlsx'
with pd.ExcelWriter(output_file) as writer:
    for cluster_id in range(optimal_k):
        cluster_data = df[df['cluster'] == cluster_id][['steamId', 'name', 'ScoreGap', 'price', 'reviewScore']]
        cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster_id}', index=False)

print(f"输出成功，文件名为：{output_file}")