import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.decomposition import PCA



s = pd.read_csv('GamesalesdataV3.csv')
X = s.values
s1 = pd.read_csv('GamessalesTarget.csv')
y = s1.values








# In[55]:


# iris = load_iris()
# X = iris.data
# y = iris.target
print ('type X')
print(type(X))
print ('type Y')
print(type(y))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)



print ('X_train, Y_train')
print(X_train, y_train)

# ##############以下是Train/plot过程##################
#
# params = {
#     'eta': 0.01,
#     'n estimators': 360,
#     'learning rate': 0.1,
#     'max_depth': 2,
#     'objective': 'multi:softprob',
#     'gamma': 0.09,
#     'lambda': 1,
#     'alpha': 0,
#     'subsample': 0.6,
#     'colsample_bytree': 0.3,
#     'min_child_weight': 4,
#     'silent': 1,
#     'seed': 1000,
#     'nthread': -1,
#     'num_class': 5
#     }
#
# # In[64]:
#
#
# plst = params.items()
#
#
# # In[65]:
#
# print('params:')
# print(params)
# print('plst:')
# print(plst)
#
#
# # In[68]:
#
#
#
# dtrain = xgb.DMatrix(X_train, y_train)
#
#
# num_rounds = 1500
# model = xgb.train(plst, dtrain, num_rounds)
#
#
#
# dtest = xgb.DMatrix(X_test)
# ans = model.predict(dtest)
# print('ans:')
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
#
# model.save_model('0001.model')
# model.dump_model('dump.raw.txt')
# dtest.save_binary('dtest.buffer')
# bst2 = xgb.Booster(model_file='0001.model')

##############以下是GridSearch/fit调参过程##################

xgb_model = xgb.XGBClassifier(objective="multi:softmax", nthread=-1, num_class=7, seed=1000)

optimized_GBM = GridSearchCV(
    xgb_model,
    {
        # 'n_estimators': np.linspace(100, 2000, 20, dtype=int),
        # 'n_estimators': np.linspace(1250, 1350, 11, dtype=int),
        'n_estimators': [1300],
        # 'max_depth': np.linspace(1, 10, 10, dtype=int),
        # 'min_child_weight': np.linspace(1, 10, 10, dtype=int),
        'max_depth': [4],
        'min_child_weight': [1],
        # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
        'max_delta_step': [0.2],
        # 'gamma': np.linspace(0, 1, 11),
        # 'gamma': np.linspace(0, 0.2, 21),
        'gamma': [0.16],
        # 'subsample': np.linspace(0, 1, 11),
        # 'colsample_bytree': np.linspace(0, 1, 11)[1:],
        'subsample': [0.9],
        'colsample_bytree': [0.7],
        # 'reg_lambda': np.linspace(0, 10, 11),
        # 'reg_alpha': np.linspace(0, 10, 11),
        'reg_lambda': [1],
        'reg_alpha': [0],
        # 'eta': np.logspace(-2, 0, 10),
        'eta': [0.01],
        # 'scale_pos_weight': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'scale_pos_weight': [0],
        # 'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2],
        'learning_rate': [0.1]
    },
    cv=3,
    verbose=5,
    n_jobs=-1,
    refit=True,
    scoring='accuracy'
)

# estimator = PCA(n_components=50)   # 使用PCA将原64维度图像压缩为20个维度
# pca_X_train = estimator.fit_transform(X_train)   # 利用训练特征决定20个正交维度的方向，并转化原训练特征
# pca_X_test = estimator.transform(X_test)
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('模型最佳交叉验证准确率: %.2f%%' % (optimized_GBM.best_score_*100))

y_pred = optimized_GBM.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("测试集准确率: %.2f%%" % (accuracy*100.0))

# xgb_model2 =  xgb.XGBClassifier(objective="multi:softmax",
#                                 nthread=-1,
#                                 num_class=5,
#                                 seed=1000,
#                                 learning_rate=0.1,
#                                 eta=0.01,
#                                 n_estimators=310,
#                                 max_depth=5,
#                                 min_child_weight=2,
#                                 max_delta_step=2,
#                                 gamma=0.14,
#                                 subsample=1,
#                                 colsample_bytree=0.6,
#                                 reg_lambda=1,
#                                 reg_alpha=0,
#                                 scale_pos_weight=0)
# xgb_model2.fit(X_train, y_train)
# plot_importance(xgb_model2)
# # y_pred2 = xgb_model2.predict(X_test)
# # accuracy2 = accuracy_score(y_test,y_pred2)
# # print("accuracy: %.2f%%" % (accuracy2*100.0))
# plt.show()
# # fit_pred = optimized_GBM.predict(X_test)
# # print('Fit_pred:')
# # print (fit_pred)


dtest2 = pd.read_csv('gametestdata.csv')
dtest2 = dtest2.values


print('预测结果:')
test_pred = optimized_GBM.predict(dtest2)
print (test_pred)