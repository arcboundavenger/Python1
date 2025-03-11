import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取 Excel 文件
file_path = 'Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx'
df = pd.read_excel(file_path, sheet_name='Steam Games Duplicate')

# 选择所需列
variables = [
    'Revenue', 'DateGap', 'PeakCCU', 'Price',
    'MetacriticScore', 'AveragePlaytime', 'MedianPlaytime',
    'TwitchPeakViewer', 'TwitchPeakChannel', 'TotalReviews',
    'UserScore', 'Follower'
]

# 过滤数据，去除缺失值
df_filtered = df[variables].dropna()

# 在过滤后的数据上添加常数项
X = sm.add_constant(df_filtered)

# 计算 VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 输出 VIF 结果
print(vif_data)