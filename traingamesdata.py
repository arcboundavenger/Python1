import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV

s = pd.read_csv(r'C:\Users\arcbo\PycharmProjects\Python1\GamesalesdataV3.csv')
X = s.values
s1 = pd.read_csv(r'C:\Users\arcbo\PycharmProjects\Python1\GamessalesTarget.csv')
y = s1.values








# In[55]:


# iris = load_iris()
# X = iris.data
# y = iris.target
print ('type X')
print(type(X))
print ('type Y')
print(type(y))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



print ('X_train, Y_train')
print(X_train, y_train)


params = {
    'eta': 0.01,
    'n estimators':620,
    'learning rate': 0.01,
    'max_depth': 3,
    'objective': 'multi:softprob',
    'gamma': 0.02,
    'max_depth': 3,
    'lambda': 1,
    'alpha':0,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'silent': 1,
    'seed': 1000,
    'nthread': -1,
    'num_class': 4
    }

# In[64]:


plst = params.items()


# In[65]:

print('params:')
print(params)
print('plst:')
print(plst)


# In[68]:



dtrain = xgb.DMatrix(X_train, y_train)


num_rounds = 5000
model = xgb.train(plst, dtrain, num_rounds)



dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
print('ans:')
print(ans)

plot_importance(model)


preds = model.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
plt.show()

#

# xgb_model = xgb.XGBClassifier(objective="multi:softprob", nthread=-1, num_class=4, seed=1000)
#
# optimized_GBM = GridSearchCV(
#     xgb_model,
#     {
#         'n_estimators': [620],
#         'max_depth': [3],
#         'min_child_weight': [1],
#         'gamma': [0.02],
#         'subsample': [0.9],
#         'colsample_bytree': [0.8],
#         'reg_lambda': [1],
#         'reg_alpha': [0],
#         'eta': [0.01],
#         'scale_pos_weight': [0.1],
#         'learning_rate': [0.01]
#     },
#     cv=5,
#     verbose=5,
#     n_jobs=-1,
#     refit=True
# )
#
#
# #
# #
# # model = xgb.XGBRegressor(**other_params)
# # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(X_train, y_train)
# evalute_result = optimized_GBM.cv_results_
# # print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))