import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'Steam_all_premium_games_detailed_all.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 处理年份数据
df['Release'] = pd.to_datetime(df['Release'])
df['Year'] = df['Release'].dt.year

# 按年份分组并计数
year_counts = df['Year'].value_counts().sort_index()

# 创建年份柱状图
def plot_yearly_releases(year_counts):
    plt.figure(figsize=(10, 6))
    bars = year_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Games Released by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # 添加数字标签
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 int(bar.get_height()), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

plot_yearly_releases(year_counts)

# 创建价格区间
bins = [0, 5, 10, 20, 40, float('inf')]
labels = ['0-4.99', '5-9.99', '10-19.99', '20-39.99', '40+']
df['Price Group'] = pd.cut(df['Price'], bins=bins, labels=labels, right=False)

# 按价格分组并计数
price_counts = df['Price Group'].value_counts().sort_index()

# 创建价格饼图
def plot_price_distribution(price_counts):
    plt.figure(figsize=(10, 6))
    plt.pie(price_counts, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

    # 添加数字标签
    for i, count in enumerate(price_counts):
        angle = (i * 360 / len(price_counts)) + (360 / (2 * len(price_counts)))
        plt.text(angle, 0, count, ha='center', va='center', fontsize=10)

    plt.title('Distribution of Games by Price (Unit: $)')
    plt.legend(labels=price_counts.index, title='Price Groups', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

plot_price_distribution(price_counts)

# 创建收入区间
revenue_bins = [0, 1000, 5000, 10000, 100000, 1000000, float('inf')]
revenue_labels = ['<1k', '1-5k', '5-10k', '10-100k', '100k-1m', '1m+']
df['Revenue Group'] = pd.cut(df['Revenue'], bins=revenue_bins, labels=revenue_labels, right=False)

# 按收入分组并计数
revenue_counts = df['Revenue Group'].value_counts().sort_index()

# 创建收入饼图
def plot_revenue_distribution(revenue_counts):
    plt.figure(figsize=(10, 6))
    plt.pie(revenue_counts, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

    # 添加数字标签
    for i, count in enumerate(revenue_counts):
        angle = (i * 360 / len(revenue_counts)) + (360 / (2 * len(revenue_counts)))
        plt.text(angle, 0, count, ha='center', va='center', fontsize=10)

    plt.title('Distribution of Games by Revenue (Unit: $)')
    plt.legend(labels=revenue_counts.index, title='Revenue Groups', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

plot_revenue_distribution(revenue_counts)

# 数据清洗：删除 Score 为 100 和 0，删除 Reviews 小于 10 的游戏
filtered_df = df[(df['Score'] != 100) & (df['Score'] != 0) & (df['Reviews'] >= 10)]

# 标记删除的部分
not_enough_reviews = df[(df['Score'] == 100) | (df['Score'] == 0) | (df['Reviews'] < 10)]
not_enough_reviews['Score'] = 'Not Enough Reviews'

# 重新定义 Score 区间
score_bins = [-1, 19, 39, 69, 79, 89, 100]  # 更新为新的区间
score_labels = ['0-19', '20-39', '40-69', '70-79', '80-89', '90+']
filtered_df['Score Group'] = pd.cut(filtered_df['Score'], bins=score_bins, labels=score_labels)

# 统计各 Score 区间的数量
score_counts = filtered_df['Score Group'].value_counts()

# 添加 'Not Enough Reviews' 的计数
not_enough_counts = not_enough_reviews['Score'].value_counts()
score_counts = pd.concat([score_counts, not_enough_counts])

# 创建 Score 的饼图
def plot_score_distribution(score_counts):
    plt.figure(figsize=(10, 6))
    plt.pie(score_counts, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

    # 添加数字标签
    for i, count in enumerate(score_counts):
        angle = (i * 360 / len(score_counts)) + (360 / (2 * len(score_counts)))
        plt.text(angle, 0, count, ha='center', va='center', fontsize=10)

    plt.title('Distribution of Game Scores')
    plt.legend(labels=score_counts.index, title='Score Groups', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

plot_score_distribution(score_counts)