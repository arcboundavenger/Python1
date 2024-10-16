import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'Gen9_5000.xlsx'  # 替换为您的文件路径
df = pd.read_excel(file_path, sheet_name='All')

# 筛选 EA? 列为 1 和 0 的数据
df_ea_0 = df[df['EA?'] == 0]
df_ea_1 = df[df['EA?'] == 1]

# 设置图形的大小
plt.figure(figsize=(18, 6))

# 绘制 Review Score 的箱线图
plt.subplot(1, 3, 1)
plt.boxplot([df_ea_0['reviewScore'], df_ea_1['reviewScore']], labels=['EA=0', 'EA=1'])
plt.title('Review Score by EA Status')
plt.ylabel('Scores')
plt.grid(axis='y')
# 计算并标注平均值
means = [df_ea_0['reviewScore'].mean(), df_ea_1['reviewScore'].mean()]
for i, mean in enumerate(means):
    plt.plot(i + 1, mean, marker='o', color='red', label='Mean' if i == 0 else "")
plt.legend()

# 绘制 Chinese Review Score 的箱线图
plt.subplot(1, 3, 2)
plt.boxplot([df_ea_0['ChineseReviewScore'], df_ea_1['ChineseReviewScore']], labels=['EA=0', 'EA=1'])
plt.title('Chinese Review Score by EA Status')
plt.ylabel('Scores')
plt.grid(axis='y')
# 计算并标注平均值
means = [df_ea_0['ChineseReviewScore'].mean(), df_ea_1['ChineseReviewScore'].mean()]
for i, mean in enumerate(means):
    plt.plot(i + 1, mean, marker='o', color='red', label='Mean' if i == 0 else "")
plt.legend()

# 绘制 English Review Score 的箱线图
plt.subplot(1, 3, 3)
plt.boxplot([df_ea_0['EnglishReviewScore'], df_ea_1['EnglishReviewScore']], labels=['EA=0', 'EA=1'])
plt.title('English Review Score by EA Status')
plt.ylabel('Scores')
plt.grid(axis='y')
# 计算并标注平均值
means = [df_ea_0['EnglishReviewScore'].mean(), df_ea_1['EnglishReviewScore'].mean()]
for i, mean in enumerate(means):
    plt.plot(i + 1, mean, marker='o', color='red', label='Mean' if i == 0 else "")
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()