import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取Excel文件中的指定工作表
df = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games Duplicate 2')

# 选择用于聚类的特征
features = df[['Revenue', 'DateGap', 'PeakCCU', 'Price', 'MetacriticScore', 'TwitchPeakViewer', 'TwitchPeakChannel', 'TotalReviews', 'UserScore', 'Follower']]

# 处理缺失值：可以选择删除或填充
# 方法1: 删除包含缺失值的行
features = features.dropna()

# 方法2: 填充缺失值（例如，用均值填充）
# features = features.fillna(features.mean())

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 肘部法选择聚类数量
inertia = []
K_range = range(1, 11)  # 从1到10聚类
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid()
plt.show()

# 选择聚类数量，例如选择3
optimal_k = 3

# 进行K-means聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_

# 雷达图函数
def radar_chart(data, title, labels):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))  # 闭合图形
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='red', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)

# 画出每个聚类的雷达图
for i in range(optimal_k):  # 根据选择的聚类数量调整
    cluster_data = df[df['Cluster'] == i].mean().values[:-1]  # 计算平均值
    radar_chart(cluster_data, f'Cluster {i}', features.columns)

plt.show()