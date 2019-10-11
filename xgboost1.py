#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt


# In[55]:


# read in the iris data
iris = load_iris()
X = iris.data
y = iris.target
print(type(X))
print(type(y))


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)


# In[57]:


print(X_train, y_train)


# In[63]:


params={    
    'booster':'gbtree',
    'objective': 'reg:gamma',
#     'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    }


# In[64]:


plst = params.items()


# In[65]:


print(params)
print(plst)


# In[68]:



dtrain = xgb.DMatrix(X_train, y_train)


num_rounds = 500
#这里直接使用 params取代 plst也可以
model = xgb.train(plst, dtrain, num_rounds)



dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
print(ans)


# In[67]:


# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.4f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()


# In[27]:





# In[ ]:




