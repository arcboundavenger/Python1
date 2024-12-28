import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games 2024')

# 检查缺失值和无穷大
print("缺失值统计：")
print(data.isnull().sum())
print("\n无穷大统计：")
print((data == float('inf')).sum())
print((data == float('-inf')).sum())

# 处理无穷大：将无穷大替换为 NaN
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

# 删除任何包含缺失值的行
data.dropna(inplace=True)

# 再次检查数据
print("\n处理后的缺失值统计：")
print(data.isnull().sum())
print("\n处理后的无穷大统计：")
print((data == float('inf')).sum())
print((data == float('-inf')).sum())

# 选择目标变量和特征
y = data['LnRevenue']
X = data.drop(columns=['LnRevenue', 'AppID', 'Estimated owners', 'Release date'])

# 添加常数项（截距）
X = sm.add_constant(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建模型并拟合
model = sm.OLS(y_train, X_train).fit()

# 打印模型摘要
print("\n模型摘要：")
print(model.summary())