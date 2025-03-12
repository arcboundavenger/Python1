import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'Region': ['Canada', 'United States', 'United Kingdom', 'China', 'Rest of the World'],
    '2024': [1048, 9113, 549, 205, 2788],
    '2023': [723, 6134, 295, 1220, 2173],
    '2022': [235, 2098, 34, 2882, 3300]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 将数据从宽格式转换为长格式
df_long = df.melt(id_vars='Region', var_name='Year', value_name='Value')

# 按年份和地区分组并绘制横向堆叠柱状图
# 指定地区的顺序
region_order = ['United States', 'Canada', 'United Kingdom', 'China', 'Rest of the World']

# 按指定顺序重新排列数据
df_long['Region'] = pd.Categorical(df_long['Region'], categories=region_order, ordered=True)
df_long = df_long.sort_values('Region')

# 按年份和地区分组
df_grouped = df_long.groupby(['Year', 'Region'])['Value'].sum().unstack()

# 按年份顺序排列
df_grouped = df_grouped.loc[['2022', '2023', '2024']]

# 定义更美观的颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 蓝、橙、绿、红、紫

# 绘制横向堆叠柱状图
ax = df_grouped.plot(kind='barh', stacked=True, figsize=(12, 6), color=colors)

# 反转 y 轴
ax.invert_yaxis()

# 设置标题和坐标轴标签
plt.ylabel('Year')

# 保存图像，dpi=1200
output_file = 'stacked_barh_by_year_and_region_ordered.png'
plt.savefig(output_file, dpi=1200, bbox_inches='tight')

# 显示图表
plt.show()