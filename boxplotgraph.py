import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X = pd.read_csv('Boxplot1.csv')
df = X #先生成0-1之间的5*4维度数据，再装入4列DataFrame中
df.boxplot() #也可用plot.box()
plt.ylim(0,500000)
plt.show()