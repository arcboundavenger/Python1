# 导入必要的库
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx', sheet_name='Steam Games 2024')

# 处理缺失值和无穷大
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# 选择目标变量和特征
y = data['LnRevenue']
X = data.drop(columns=['LnRevenue', 'AppID', 'Estimated owners', 'Release date'])
X = sm.add_constant(X)  # 添加常数项

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 拟合模型
model = sm.OLS(y_train, X_train).fit()

# 打印模型摘要
print(model.summary())

# 计算预测值和评估指标
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
nrmse = rmse / (y_test.max() - y_test.min())
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}, NRMSE: {nrmse}, MAE: {mae}")

# 输出 R² 和 F 值
r_squared = model.rsquared
f_value = model.fvalue
print(f"R²: {r_squared}, F: {f_value}")

# 计算残差
residuals = y_test - y_pred

# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# 绘制预测误差图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Prediction Error Plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid()
plt.show()

# 处理 p 值和显著性以及其他统计信息
conf_int = model.conf_int()  # 获取置信区间
results_df = pd.DataFrame({
    'Parameter': model.params.index,
    'Estimate (coef)': model.params.values,
    'Std Err': model.bse.values,
    't': model.tvalues.values,
    'P-value': model.pvalues.values,
    'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in model.pvalues],
    'LLCI': conf_int[0].values,  # 下限
    'ULCI': conf_int[1].values,  # 上限
    'R²': r_squared,              # R²
    'F': f_value                  # F 值
})

# 保存结果
results_df.to_excel('model_results.xlsx', index=False)
print("模型结果已保存为 'model_results.xlsx'.")