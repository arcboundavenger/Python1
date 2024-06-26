
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score




raw_data=pd.read_excel('1111.xlsx', engine='openpyxl')
data=raw_data.iloc[:]

# 数据归一化
min_max_scale=MinMaxScaler()
data=min_max_scale.fit_transform(data)

# distortions = []
# K = range(1,10)
# for k in K:
#     kmodel = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=10, random_state=1)
#     kmodel.fit(data)
#     distortions.append(kmodel.inertia_)
#
# plt.figure()
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()




##################################################
# #调用k-means算法，进行聚类分析
k = 5 # 定义聚类个数
kmodel = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=15, tol=0.00001, random_state=10).fit(data)
#查看聚类中心
# print(kmodel.cluster_centers_)
#查看各样本对应的类别
# print(kmodel.labels_ )
print(kmodel.inertia_)

#####################
# 绘制雷达图
#标签
# labels = raw_data.columns
labels = np.array(['Sales','Youtube',
                   'TwitchPeakChannel',
                   'TwitchPeakViewer',
                   'MCUserRatings',
                   'Douban',
                   'GoogleTrends','Sales']) #第一个和最后一个永远一样
print(labels)

#每个类别中心点数据
plot_data = kmodel.cluster_centers_
#指定颜色
# color = ['b', 'g', 'r', 'c', 'y']
# 设置角度
j=7 #特征的个数
angles = np.linspace(0, 2*np.pi, j, endpoint=False)

# 闭合
angles = np.concatenate((angles, [angles[0]]))
plot_data = np.concatenate((plot_data, plot_data[:,[0]]), axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, polar=True) # polar参数为True即极坐标系
for i in range(len(plot_data)):
    ax.plot(angles, plot_data[i], 'o-', label = str(i), linewidth=3)

# ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5), fontproperties="SimHei") # 手动配置r网格刻度
ax.set_thetagrids(angles * 180/np.pi, labels)
# ax.set_ylim([0, 1])
plt.legend(loc = 4) # 设置图例位置
leg = ax.legend(fontsize = 'large')
leg.set_title('Group', prop = {'size':'x-large'})
plt.xticks(fontsize = 16)

plt.show()

#################################

r1 = pd.Series(kmodel.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(kmodel.cluster_centers_) #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
print(r)


#详细输出原始数据及其类别
r = pd.concat([pd.DataFrame(data), pd.Series(kmodel.labels_)], axis = 1)  #详细输出每个样本对应的类别
r.to_csv('1112.csv') #保存结果
