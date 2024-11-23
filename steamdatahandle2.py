import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'Steam_all_premium_games_detailed_all.xlsx'  # 使用您提供的文件名
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 确保日期列存在，使用 'Release' 列表示游戏发布的日期
df['Release'] = pd.to_datetime(df['Release'])  # 转换为 datetime 格式

# 添加年份列
df['year'] = df['Release'].dt.year

# 筛选 2014 到 2024 年的数据
df = df[(df['year'] >= 2003) & (df['year'] <= 2024)]

# 生成从2014年到2024年的年份列表
years = pd.Series(range(2003, 2025))

# 初始化一个新的 DataFrame 来存储累计数据
cumulative_data = pd.DataFrame(columns=['year', 'Sub-Genre', 'appid_count', 'revenue_mean'])

for year in years:
    # 计算截至当前年份的累计数据
    cumulative_year_data = df[df['year'] <= year].groupby('Sub-Genre').agg(
        appid_count=('appid', 'count'),
        revenue_mean=('Revenue', 'mean')
    ).reset_index()

    cumulative_year_data['year'] = year  # 添加年份信息
    cumulative_data = pd.concat([cumulative_data, cumulative_year_data], ignore_index=True)

# 数据处理
subgenre_agg = df.groupby('Sub-Genre').agg(
    appid_count=('appid', 'count'),
    revenue_mean=('Revenue', 'mean'),
    revenue_sum=('Revenue', 'sum')
).reset_index()

# 计算辅助线的值
revenue_upper_quartile = subgenre_agg['revenue_mean'].quantile(0.75)  # 收入的上四分位数
appid_count_lower_quartile = subgenre_agg['appid_count'].quantile(0.25)  # appid_count 的下四分位数

# 创建一个新的图形用于绘制折线图
plt.figure(figsize=(9, 9))  # 设置图形尺寸为正方形

# 选择要绘制的 subgenre 列表
subgenres_to_plot = [
    'Asynchronous Multiplayer',
    'Life Sim',
    'Roguelike Deckbuilder'
]

# 绘制每个指定 subgenre 的折线图
for subgenre in subgenres_to_plot:
    subgenre_data = cumulative_data[cumulative_data['Sub-Genre'] == subgenre]

    # 只选择有效的（非零）数据以避免对数坐标问题
    valid_data = subgenre_data[(subgenre_data['appid_count'] > 0) & (subgenre_data['revenue_mean'] > 0)]

    if not valid_data.empty:  # 检查有效数据
        plt.plot(valid_data['appid_count'], valid_data['revenue_mean'], marker='o', label=subgenre)

# 设置对数坐标
plt.xscale('log')  # 设置横轴为对数坐标
plt.yscale('log')  # 设置纵轴为对数坐标

# 设置图形细节
plt.xlabel('Number of Games')
plt.ylabel('Average Revenue (Unit: $)')
plt.grid(True)

# 添加图例，放在左下角
plt.legend(title='Sub-Genres', loc='lower left', bbox_to_anchor=(0.0, 0.0))

# 设置坐标轴为正方形
plt.axis('equal')

# 设置坐标轴范围
plt.xlim(10 ** 0, 10 ** 4)  # 设置横轴范围
plt.ylim(10 ** 4, 10 ** 8)  # 设置纵轴范围

# 显示图形
plt.show()