
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score
import xgboost as xgb
from xgboost import plot_importance

X = pd.read_csv('GamesalesdataVSteam.csv')
y = pd.read_csv('GamessalesTargetVSteam.csv')

file = open('GamesalesdataV3.csv', 'r')
lines = file.readlines()
feature_name_list = lines[0].split(",")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##############以下是GridSearch/fit调参过程##################

n_estimators = [100, 200, 500, 800]
max_depth = [3, 4, 5, 6]
learning_rate = [0.01, 0.1, 0.2]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)

regress_model = xgb.XGBRegressor(objective="reg:squarederror", nthread=-1, seed=1000)
gs = GridSearchCV(regress_model, param_grid, verbose=1, refit=True, scoring='r2', cv=3, n_jobs=-1)
gs.fit(X_train, y_train)  # X为训练数据的特征值，y为训练数据的label
# 性能测评
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型R2得分:", gs.best_score_)

xgb_model2 = gs.best_estimator_
y_pred = xgb_model2.predict(X_test)
y_pred_len = len(y_pred)
data_arr=[]
y_tests=np.array(y_test)

print("最佳测试集R2得分:",r2_score(y_tests,y_pred))

for row in range(0, y_pred_len):
        data_arr.append([y_tests[row][0], y_pred[row]])
np_data = np.array(data_arr)
pd_data = pd.DataFrame(np_data, columns=['y_test', 'y_predict'])
plot_importance(xgb_model2, importance_type='weight')
plt.show()