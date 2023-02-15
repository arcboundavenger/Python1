#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[55]:


# read in the iris data
iris = load_iris()
X = iris.data
y = iris.target
print ('type X')
print(type(X))
print ('type Y')
print(type(y))


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[57]:

print ('X_train, Y_train')
print(X_train, y_train)


# In[63]:


# params={
#     'booster':'gbtree',
#     'objective': 'reg:gamma',
# #     'num_class': 3,
#     'gamma': 0.1,
#     'max_depth': 6,
#     'lambda': 2,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'min_child_weight': 3,
#     'silent': 1,
#     'eta': 0.1,
#     'seed': 1000,
#     'nthread': 4,
#     }
params = {
    'booster':'gbtree',
    'eta': 1,
    'max_depth': 3,
    'objective': 'multi:softprob',
    'gamma': 0.1,
    'max_depth': 7,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': -1,
    'num_class': 3
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


num_rounds = 500
#这里直接使用 params取代 plst也可以
model = xgb.train(plst, dtrain, num_rounds)



dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
print('ans:')
print(ans)


# In[67]:

# 计算准确率
# cnt1 = 0
# cnt2 = 0
# for i in range(len(y_test)):
#     if ans[i] == y_test[i]:
#         cnt1 += 1
#     else:
#         cnt2 += 1
# print("Accuracy: %.4f %% " % (100 * cnt1 / (cnt1 + cnt2)))
#

# 显示重要特征
plot_importance(model)
# plt.show()

preds = model.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
plt.show()

# clf = xgb.XGBClassifier()
# parameters = {
#     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#     "min_child_weight" : [ 1, 3, 5, 7 ],
#     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
# }
# grid = GridSearchCV(clf,
#                     parameters, n_jobs=4,
#                     scoring="neg_log_loss",
#                     cv=3)
# grid.fit(X_train, y_train)
# m.dump_model('dump.raw.txt')




