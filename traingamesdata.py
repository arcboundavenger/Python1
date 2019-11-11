import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV

#读游戏的csv表
s = pd.read_csv('GamesalesdataV3.csv')
X = s.values
s1 = pd.read_csv('GamessalesTarget.csv')
y = s1.values

#鸢尾花：
# iris = load_iris()
# X = iris.data
# y = iris.target
print(type(X))
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print ('X_train, Y_train')
print(X_train, y_train)

#
# params = {
#     'eta': 0.01,
#     'n estimators':900,
#     'learning rate': 0.01,
#     'max_depth': 5,
#     'objective': 'multi:softprob',
#     'gamma': 0.04,
#     'lambda': 10,
#     'alpha':0,
#     'subsample': 0.9,
#     'colsample_bytree': 0.8,
#     'min_child_weight': 1,
#     'silent': 1,
#     'seed': 1000,
#     'nthread': -1,
#     'num_class': 6
#     }
#
# plst = params.items()
# print(params)
# print(plst)
#
# dtrain = xgb.DMatrix(X_train, y_train)
#
#
# num_rounds = 900
# model = xgb.train(plst, dtrain, num_rounds)
#
# dtest = xgb.DMatrix(X_test)
# ans = model.predict(dtest)
# print(ans)
#
# plot_importance(model)
#
#
# preds = model.predict(dtest)
# best_preds = np.asarray([np.argmax(line) for line in preds])
# print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
# print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
# print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
# plt.show()

# 下面是GridSearch调参：
xgb_model = xgb.XGBClassifier(objective="multi:softmax", nthread=-1, num_class=5, seed=1000)

optimized_GBM = GridSearchCV(
    xgb_model,
    {
        'n_estimators': [620],
        'max_depth': [7],
        'min_child_weight': [3],
        'gamma': [0.2],
        'subsample': [0.8],
        'colsample_bytree': [0.6],
        'reg_lambda': [2],
        'reg_alpha': [0],
        'eta': [0.01],
        'scale_pos_weight': [0],
        'learning_rate': [0.01]
    },
    cv=5,
    verbose=5,
    n_jobs=-1,
    refit=True,
    scoring='accuracy'
)


#
#
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
