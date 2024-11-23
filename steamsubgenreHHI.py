import pandas as pd

# 读取 Excel 文件
file_path = 'Steam_all_premium_games_detailed_all.xlsx'
xls = pd.ExcelFile(file_path)

# 查看所有工作表的名称
print("工作表名称:", xls.sheet_names)

# 读取特定的工作表，例如 'Sheet1'
df = pd.read_excel(xls, sheet_name='Sheet1')

# 显示数据的前几行
print("数据预览:")
print(df.head())

# 检查数据的信息
print("\n数据基本信息:")
print(df.info())

# 确保数据中有需要的列
required_columns = ['Sub-Genre', 'Revenue']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"数据中缺少必要列: {required_columns}")

# 计算每个 Sub-Genre 的总收入
subgenre_revenue = df.groupby('Sub-Genre')['Revenue'].sum().reset_index()
subgenre_revenue.rename(columns={'Revenue': 'Total Revenue'}, inplace=True)

# 合并原始数据与总收入
df = df.merge(subgenre_revenue, on='Sub-Genre')

# 计算每款游戏的收入占该 Sub-Genre 的百分比
df['Revenue Percentage'] = df['Revenue'] / df['Total Revenue']

# 计算 HHI
df['HHI Contribution'] = df['Revenue Percentage'] ** 2
hhi = df.groupby('Sub-Genre')['HHI Contribution'].sum().reset_index()

# 计算 HHI，乘以 10000 转换为常用单位
hhi['HHI'] = hhi['HHI Contribution'] * 10000

# 输出 HHI 结果
print("\n每个 Sub-Genre 的 HHI:")
print(hhi[['Sub-Genre', 'HHI']])

# 将结果保存到新的 Excel 文件
output_file_path = 'SubGenre_HHI.xlsx'
hhi[['Sub-Genre', 'HHI']].to_excel(output_file_path, index=False)

print(f"\nHHI 结果已保存到 {output_file_path}")