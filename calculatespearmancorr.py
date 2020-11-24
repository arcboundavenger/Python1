
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import palettable


# train = pd.read_csv('forspearman.csv')
train = pd.read_csv('forpearson.csv')

dcorr = train.corr(method='spearman')#默认为'pearson'检验，可选'kendall','spearman'
plt.figure(figsize=(11, 9),dpi=100)
plt.title('Spearman Correlation of Features')
# colormap = plt.cm.viridis
sns.heatmap(data=dcorr,
            # cmap=colormap,
            linewidths=0.1, vmax=1.0 ,fmt=".2f", square=True, annot=True,annot_kws={'size':8,'weight':'normal', 'color':'white'},mask=np.triu(np.ones_like(dcorr,dtype=np.bool)))
plt.show()