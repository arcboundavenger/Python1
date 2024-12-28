import pandas as pd
from lightgbm import LGBMRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games 2024')

# 选择目标变量和特征
y = data['LnTotalReviews']
# 删除不需要的列
X = data.drop(columns=['LnTotalReviews', 'AppID', 'Estimated owners', 'Release date'])

# 确保所有数据都是数值类型
X = X.apply(pd.to_numeric, errors='coerce')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LGBMRegressor()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型使用 R^2
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# 绘制特征的重要性
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='split', max_num_features=10)
plt.title('Feature Importance')
plt.show()