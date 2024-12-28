import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 1. 加载加州住房数据集
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练LightGBM模型
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# 4. Gain和Split重要性
lgbm_gain_importance = model.feature_importances_  # Gain重要性
lgbm_split_importance = model.feature_importances_  # Split重要性

# 5. SHAP值
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 6. Permutation重要性
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# 7. 可视化结果
plt.figure(figsize=(15, 10))

# Gain重要性
plt.subplot(2, 2, 1)
plt.barh(X.columns, lgbm_gain_importance, color='skyblue')
plt.title('Gain Importance')
plt.xlabel('Importance')

# Split重要性
plt.subplot(2, 2, 2)
plt.barh(X.columns, lgbm_split_importance, color='lightgreen')
plt.title('Split Importance')
plt.xlabel('Importance')

# Permutation重要性
plt.subplot(2, 2, 3)
plt.barh(X.columns, perm_importance.importances_mean, color='salmon')
plt.title('Permutation Importance')
plt.xlabel('Importance')

# SHAP值可视化
plt.subplot(2, 2, 4)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Importance')

plt.tight_layout()
plt.show()