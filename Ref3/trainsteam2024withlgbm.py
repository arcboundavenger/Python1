import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel("Steam Games 2024_Filled with MC and Twitch_used for Analysis.xlsx")

# 2. 数据清理，删除不需要的列
data.drop(columns=["AppID", "Estimated owners", "Release date"], inplace=True)

# 3. 标签编码
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# 4. 特征和目标
X = data.drop(columns=["Class"])
y = data["Class"]

# 5. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 设置LightGBM参数范围
param_grid = {
    'num_leaves': [31, 63],
    'max_depth': [-1, 10],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200]
}

# 7. 创建LightGBM模型
lgb_model = lgb.LGBMClassifier(objective='multiclass')

# 8. 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid,
                           scoring='f1_weighted', cv=5, verbose=1)

# 9. 训练模型
grid_search.fit(X_train, y_train)

# 10. 输出最佳参数和最佳得分
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# 11. 使用最佳参数模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 12. 输出评估结果
print(classification_report(y_test, y_pred))

# 13. 输出特征重要性
importance = best_model.feature_importances_
feature_names = X.columns

# 创建特征重要性数据框
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # 反转y轴，使最重要的特征在上面
plt.show()