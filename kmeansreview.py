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

# 过滤掉包含 NaN 的行
df = df.dropna(subset=['ScoreGap', 'price'])

# 选择要聚类的特征
X = df[['ScoreGap', 'price']]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用轮廓系数确定最佳聚类数
silhouette_scores = []
k_range = range(2, 11)  # 从2到10个聚类
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# 绘制轮廓系数图
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# K-means 聚类（假设选择的聚类数为3）
optimal_k = 3  # 根据轮廓系数图选择聚类数
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 可视化聚类结果
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

# 显示图形
plt.show()