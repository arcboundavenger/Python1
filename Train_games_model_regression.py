import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle



X = pd.read_csv('GamesalesdataV3.csv') #这个是x值表
y = pd.read_csv('GamessalesTargetV2.csv') #这个是y值表




file = open('GamesalesdataV3.csv', 'r')
lines = file.readlines()
feature_name_list = lines[0].split(",")



print('type X')
print(type(X))
print('type Y')
print(type(y))
j=200 #随机split多少次，可以改的大一些
new_pred=np.zeros((j,20)) #20代表想要预测游戏的个数，随情况调整，我忘了先读表了，所以都是手动改的

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

        dtest2 = pd.read_csv('gametestdata.csv') #这个地方是要预测的游戏的x值列表，我给你的文件里是一些我自己取的游戏数据

        test_pred = xgb_model2.predict(dtest2)

        for i in range(0,len(test_pred)):
                new_pred[ii][i]=int(math.exp(test_pred[i])) #这里的数字是销量的对数，我在里面用math.exp还原了
pd.DataFrame(new_pred).to_csv('PredictResult.csv', index=None) #把预测数字存起来，应该是你随机了多少次，每列就是每次随机打乱模型后的预测结果；把这些预测数字求平均值和标准差，就能得到销量的区间
#
# 把模型存起来
# pickle.dump(xgb_model2, open("xgb1", "wb"))

# 读取模型
# loaded_model = pickle.load(open("xgb1", "rb"))
#
# 用读取的模型预测
# test_pred = loaded_model.predict(dtest2)
# print(test_pred)