import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'Steam_all_premium_games_detailed_all.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 设置要绘制的列
columns_to_plot = ['Price', 'Followers', 'Reviews', 'Score', 'Revenue']

# 创建箱线图
plt.figure(figsize=(12, 8))

for i, column in enumerate(columns_to_plot):
    plt.subplot(2, 3, i + 1)  # 创建2行3列的子图
    df.boxplot(column=column)
    plt.title(f'Boxplot of {column}')

    # 计算平均值并标注
    mean_value = df[column].mean()
    plt.axhline(mean_value, color='red', linestyle='--')
    plt.text(1.1, mean_value, f'Mean: {mean_value:.2f}', color='red')

plt.tight_layout()
plt.show()