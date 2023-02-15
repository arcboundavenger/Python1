from keras import layers
from keras import models
from keras.datasets import imdb
import numpy as np
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import StandardScaler  # 用于特征的标准化
from sklearn.preprocessing import Imputer


print("Loading Data ... ")

# 导入数据
X = pd.read_csv('GamesalesdataV3.csv')
X = X.values
y = pd.read_csv('GamessalesTarget.csv')
y = y.values








# In[55]:


# iris = load_iris()
# X = iris.data
# y = iris.target
print ('type X')
print(type(X))
print ('type y')
print(type(y))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

model = models.Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],)))
model.add(Activation('tanh'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dense(16))
model.add(Activation('linear'))
model.add(Dense(1)) # 这里需要和输出的维度一致
model.add(Activation('softmax'))

# For a multi-class classification problem
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_val = X_train[:50]
partial_X_train = X_train[50:]

y_val = y_train[:50]
partial_y_train = y_train[50:]

history = model.fit(partial_X_train, partial_y_train,
                      epochs=20,
                      batch_size=200,
                      validation_split=0.1,shuffle=True)



history_dict = history.history
print(history_dict.keys())

# 5. Plotting the training and validation loss

# import matplotlib.pyplot as plt
#
# # 画出训练集和验证集的损失和精度变化，分析模型状态
#
# acc = history.history['binary_accuracy']  # 训练集acc
# val_acc = history.history['val_binary_accuracy']  # 验证集 acc
# loss = history.history['loss']  # 训练损失
# val_loss = history.history['val_loss']  # 验证损失
#
# epochs = range(1, len(acc)+1)  # 迭代次数
#
# plt.plot(epochs, loss, 'bo', label='Training loss')  # bo for blue dot 蓝色点
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()  # clar figure
#
#
# plt.plot(epochs, acc, 'bo', label='Training acc')  # bo for blue dot 蓝色点
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
# # ----------------------------------------使用model 定义网络
# input_tensor = layers.Input(shape=(10000,))
# x = layers.Dense(32, activation='relu')(input_tensor)
# x = layers.Dense(16, activation='relu')(x)
# x = layers.Dense(16, activation='relu')(x)
# output_tensor = layers.Dense(1, activation='sigmoid')(x)
# network = models.Model(inputs=input_tensor, outputs=output_tensor)
#
# network.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# network.fit(x_train, y_train, epochs=4, batch_size=512)
#
# print("test data evaluate, epochs=20", m.evaluate(x_test, y_test))
# print("test data evaluate, epochs=4 ", network.evaluate(x_test, y_test))

