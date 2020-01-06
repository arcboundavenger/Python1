import csv
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import plot_importance
import pickle



X = pd.read_csv('GamesalesdataV3.csv')
y = pd.read_csv('GamessalesTarget.csv')




file = open('GamesalesdataV3.csv', 'r')
lines = file.readlines()
feature_name_list = lines[0].split(",")



print('type X')
print(type(X))
print('type Y')
print(type(y))

CVAccuracy=[]
TestAccuracy=[]


for j in range(20,21,1):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=j)

    # print('type X_train, type Y_train')
    # print(type(X_train), type(y_train))
    # print(X_train, y_train)

    ##############以下是GridSearch/fit调参过程##################

    xgb_model = xgb.XGBClassifier(objective="multi:softmax", nthread=-1, num_class=8, seed=1000)

    optimized_GBM = GridSearchCV(
        xgb_model,
        {
            # 'n_estimators': np.linspace(50, 1000, 20, dtype=int),
            # 'n_estimators': np.linspace(50, 150, 11, dtype=int),
            'n_estimators': [500],
            # 'max_depth': np.linspace(1, 10, 10, dtype=int),
            # 'min_child_weight': np.linspace(1, 10, 10, dtype=int),
            'max_depth': [5],
            'min_child_weight': [1],
            # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
            'max_delta_step': [0],
            # 'gamma': np.linspace(0, 1, 11),
            # 'gamma': np.linspace(0, 0.1, 11),
            'gamma': [0.0],
            # 'subsample': np.linspace(0, 1, 11),
            # 'colsample_bytree': np.linspace(0, 1, 11)[1:],
            'subsample': [.8],
            'colsample_bytree': [.8],
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

    print(j)
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('模型最佳交叉验证准确率: %.2f%%' % (optimized_GBM.best_score_*100))

    y_pred = optimized_GBM.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("测试集准确率: %.2f%%" % (accuracy*100.0))

#     CVAccuracy.append(optimized_GBM.best_score_*100)
#     TestAccuracy.append(accuracy*100.0)
#
# print(CVAccuracy)
# print(TestAccuracy)


xgb_model2 =  xgb.XGBClassifier(objective="multi:softmax",
                                nthread=-1,
                                num_class=8,
                                seed=1000,
                                learning_rate=0.1,
                                eta=0.01,
                                n_estimators=500,
                                max_depth=5,
                                min_child_weight=1,
                                max_delta_step=0,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_lambda=1,
                                reg_alpha=0,
                                scale_pos_weight=0)
xgb_model2.fit(X_train, y_train)
plot_importance(xgb_model2, importance_type='weight')
plt.show()

###############以下是Train/plot过程##################
#
# params = {
#     'eta': 0.01,
#     'n estimators': 500,
#     'learning rate': 0.1,
#     'max_depth': 5,
#     'objective': 'multi:softprob',
#     'gamma': 0.0,
#     'lambda': 1,
#     'alpha': 0,
#     'subsample': 1,
#     'colsample_bytree': 1,
#     'min_child_weight': 1,
#     'silent': 0,
#     'max_delta_step': 0,
#     'scale_pos_weight':0,
#     'seed': 1000,
#     'nthread': -1,
#     'num_class': 8
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
# num_rounds = 500
# model = xgb.train(plst, dtrain, num_rounds)
#
#
#
# dtest = xgb.DMatrix(X_test)
# ans = model.predict(dtest)
# # print('ans:')
# # print(ans)
#
#
# preds = model.predict(dtest)
# best_preds = np.asarray([np.argmax(line) for line in preds])
# print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
# print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
# print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
#
# model.feature_names = feature_name_list
#
# plot_importance(model, importance_type='weight')
# plt.show()
#
# model.save_model('0001.model')
# model.dump_model('dump.raw.txt')
# dtest.save_binary('dtest.buffer')
# bst2 = xgb.Booster(model_file='0001.model')
#
# #################以下是预测过程#############
#

dtest2 = pd.read_csv('gametestdata.csv')
# dtest2 = dtest2.values


# print('预测结果:')
# test_pred = optimized_GBM.predict(dtest2)
# print (test_pred)

# # save model to file
# pickle.dump(xgb_model2, open("xgb1", "wb"))
#
# # some time later...
#
# # load model from file
# loaded_model = pickle.load(open("xgb1", "rb"))
#
# # make predictions for test data
# y_pred = loaded_model.predict(X_test)
# predictions = [round(value) for value in y_pred]
