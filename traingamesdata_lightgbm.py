import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV  # Perforing grid search

print("Loading Data ... ")

# 导入数据
X = pd.read_csv('GamesalesdataV3.csv')
# X = s.values
y = pd.read_csv('GamessalesTarget.csv')
# y = s1.values








# In[55]:


# iris = load_iris()
# X = iris.data
# y = iris.target
print ('type X')
print(type(X))
print ('type y')
print(type(y))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)



print ('X_train, Y_train')
print(X_train, y_train)

# # create dataset for lightgbm
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# # specify your configurations as a dict
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'num_class': 9,
#     'metric': 'multi_error',
#     'num_leaves': 300,
#     'min_data_in_leaf': 100,
#     'learning_rate': 0.01,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'lambda_l1': 0.4,
#     'lambda_l2': 0.5,
#     'min_gain_to_split': 0.2,
#     'verbose': 5,
#     'is_unbalance': True
# }
#
# # train
# print('Start training...')
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10000,
#                 valid_sets=lgb_eval,
#                 early_stopping_rounds=500)
#
# print('Start predicting...')
#
# preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 输出的是概率结果
#
# # 导出结果
# for pred in preds:
#     result = prediction = int(np.argmax(pred))
#
# # 导出特征重要性
# importance = gbm.feature_importance()
# names = gbm.feature_name()
# with open('./feature_importance.txt', 'w+') as file:
#     for index, im in enumerate(importance):
#         string = names[index] + ', ' + str(im) + '\n'
#         file.write(string)
#


parameters = {
    # 'n_estimators': np.linspace(100, 2000, 20, dtype=int),
    # 'n_estimators': np.linspace(50, 150, 11, dtype=int),
    'n_estimators': [100],
    # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'learning_rate': [0.1],
    # 'max_depth': range(2,20,1),
    # 'num_leaves':range(5, 20, 5),
    'max_depth': [6],
    'num_leaves': [10],
    # 'min_child_samples': range(1, 20, 1),
    # 'min_child_weight': np.linspace(1, 10, 10, dtype=int),
    'min_child_samples': [19],
    'min_child_weight': [2],
    # 'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    # 'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'feature_fraction': [0.8],
    'bagging_fraction': [0.9],
    # 'bagging_freq': [2, 4, 5, 6, 8],
    'bagging_freq': [2],
    'lambda_l1': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'lambda_l2': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5, 1, 5, 10, 15, 35, 40],
    # 'lambda_l1': [0],
    # 'lambda_l2': [1],
    # 'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'min_split_gain':[0],
    # 'cat_smooth': [1, 10, 15, 20, 35],
    'cat_smooth': [1]


}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'multiclass',
                         metric = 'multi_error',
                         num_class = 5,
                         verbose = 10,
                         learning_rate = 0.01,
                         num_leaves = 300,
                         min_child_samples = 18,
                         min_child_weight = 0.001,
                         min_gain_to_split = 0.2,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(X_train, y_train)


print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
print("Best score: %0.4f" % gsearch.best_score_)

y_pred = gsearch.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("测试集准确率: %.2f%%" % (accuracy*100.0))


dtest2 = pd.read_csv('gametestdata.csv')

print('Test_pred:')
test_pred = gsearch.predict(dtest2)
print (test_pred)