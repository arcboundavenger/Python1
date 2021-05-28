import csv
import math
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, r2_score
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import plot_importance
import pickle

X = pd.read_csv('GamesalesdataV3.csv')
y = pd.read_csv('GamessalesTargetV2.csv')

file = open('GamesalesdataV3.csv', 'r')
lines = file.readlines()
feature_name_list = lines[0].split(",")

print('type X')
print(type(X))
print('type Y')
print(type(y))
j=100 #随机多少次，可以改的大一些


dtest2 = pd.read_csv('gametestdata.csv') #这个地方是要预测的游戏的x值列表
new_pred = np.zeros((j, 10))  # 代表想要预测游戏的个数，随情况调整，我忘了先读表了，所以都是手动改的

for ii in range(0,j):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=ii)


        ##############以下是GridSearch/fit调参过程##################

        n_estimators = [500]
        max_depth = [4]
        learning_rate = [0.1]
        param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
        regress_model = xgb.XGBRegressor(objective="reg:squarederror", nthread=-1, seed=1000)
        gs = GridSearchCV(regress_model, param_grid, verbose=1, refit=True, scoring='r2', cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        print(ii)
        print("参数的最佳取值：:", gs.best_params_)
        print("最佳模型得分:", gs.best_score_)

        xgb_model2 = gs.best_estimator_
        y_pred = xgb_model2.predict(X_test)
        y_pred_len = len(y_pred)
        data_arr=[]
        y_tests=np.array(y_test)
        print(r2_score(y_tests,y_pred))

        # for row in range(0, y_pred_len):
        #         data_arr.append([y_tests[row][0], y_pred[row]])
        # np_data = np.array(data_arr)
        # pd_data = pd.DataFrame(np_data, columns=['y_test', 'y_predict'])
        # pd_data.to_csv('submit.csv', index=None)
        # fig, ax = plt.subplots(figsize=(12,12))
        # # plot_importance(xgb_model2, importance_type='total_gain', ax=ax, title='Feature Importance (total_gain)', xlabel='Feature Score')
        # plot_importance(xgb_model2, importance_type='gain', ax=ax, title='Feature Importance (gain)', xlabel='Feature Score')
        # # plot_importance(xgb_model2, importance_type='weight', ax=ax, title='Feature Importance (weight)', xlabel='Feature Score')

        # plt.show()
        # r2_1 = gs.best_score_
        # r2_2 = r2_score(y_tests,y_pred)
        # if r2_1<0.88 or r2_2<0.88:
        #         print('Drop it!')
        #         continue

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
        #################以下是预测过程#############




        print('预测结果:')
        test_pred = xgb_model2.predict(dtest2)
        # print(test_pred)
        for i in range(0,len(test_pred)):
                new_pred[ii][i]=math.exp(test_pred[i]) #这里的数字是销量的对数，我在里面用math.exp还原了
        # print(new_pred)
pd.DataFrame(new_pred).to_csv('PredictResult.csv', index=None)
#




# 把模型存起来
# pickle.dump(xgb_model2, open("xgb1", "wb"))

# 读取模型
# loaded_model = pickle.load(open("xgb1", "rb"))
#
# 用读取的模型预测
# test_pred = loaded_model.predict(dtest2)
# print(test_pred)