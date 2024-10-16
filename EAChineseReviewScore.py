import pandas as pd
from statsmodels.stats.weightstats import ztest

# 读取 Excel 文件
file_path = 'Gen9_5000.xlsx'  # 替换为您的文件路径
df = pd.read_excel(file_path, sheet_name='All')

# 筛选 EA? 列为 0 和 1 的数据
df_ea_0 = df[df['EA?'] == 0]
df_ea_1 = df[df['EA?'] == 1]

# 计算平均值
mean_ea_0 = df_ea_0['ChineseReviewScore'].mean()
mean_ea_1 = df_ea_1['ChineseReviewScore'].mean()

# 执行独立样本 Z 检验
z_stat, p_value = ztest(df_ea_0['ChineseReviewScore'], df_ea_1['ChineseReviewScore'])

# 输出结果
print(f"EA=0 的平均 Chinese Review Score: {mean_ea_0:.4f}")
print(f"EA=1 的平均 Chinese Review Score: {mean_ea_1:.4f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 结果解释
alpha = 0.05
if p_value < alpha:
    print("在显著性水平 0.05 下，EA? 为 0 和 1 的 Chinese Review Score 平均值有显著差异。")
else:
    print("在显著性水平 0.05 下，EA? 为 0 和 1 的 Chinese Review Score 平均值没有显著差异。")