import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games Duplicate')

# 处理无穷大和缺失值
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# 选择所有变量，去掉指定的列
data_cleaned = data.drop(columns=['AppID', 'Estimated owners', 'Release date'])

# 检查Revenue列是否存在
if 'Revenue' not in data_cleaned.columns:
    raise ValueError("Revenue column not found in the data")

# 定义虚拟变量
dummy_columns = [
    'Czech', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Polish',
    'Portuguese', 'Portuguese-Brazil', 'Portuguese-Portugal', 'Russian',
    'SimplifiedChinese', 'Spanish-LatinAmerica', 'Spanish-Spain', 'Thai',
    'TraditionalChinese', 'OnlineCo-op', 'OnlinePvP', 'Single-player',
    'SteamAchievements', 'SteamTradingCards', 'Action', 'Adventure', 'Casual',
    'EarlyAccess', 'FreetoPlay', 'Indie', 'MassivelyMultiplayer', 'RPG',
    'Racing', 'Simulation', 'Sports', 'Strategy'
]

# 创建包含虚拟变量和收入的DataFrame
dummy_variables_with_revenue = data_cleaned[['Revenue'] + dummy_columns]

# 计算相关系数矩阵
dummy_correlation_matrix = dummy_variables_with_revenue.corr()

# 绘制虚拟变量和收入的相关系数热图
plt.figure(figsize=(12, 8))
sns.heatmap(dummy_correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Heatmap (Dummy Variables with Revenue)')
plt.show()

# 计算数值变量的相关系数矩阵
numerical_columns = data_cleaned.columns.difference(dummy_columns).tolist()

# 按照指定顺序重新排列
desired_order = [
    'DateGap', 'PeakCCU', 'Price', 'MetacriticScore',
    'AveragePlaytime', 'MedianPlaytime', 'TwitchPeakViewer',
    'TwitchPeakChannel', 'TotalReviews', 'UserScore', 'Follower', 'Revenue'
]

# 确保所有列都存在于数据中
numerical_columns = [col for col in desired_order if col in numerical_columns]

# 确保Revenue在第一位
if 'Revenue' in numerical_columns:
    numerical_columns.remove('Revenue')  # 移除Revenue
    numerical_columns.insert(0, 'Revenue')  # 在第一位插入Revenue

numerical_variables = data_cleaned[numerical_columns]

# 绘制数值变量的相关系数热图
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_variables.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Heatmap (Numerical Variables)')
plt.show()