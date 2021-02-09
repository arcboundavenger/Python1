
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import palettable


train = pd.read_csv('forspearman.csv')
# train = pd.read_csv('forpearson.csv')

dcorr = train.corr(method='spearman')#默认为'pearson'检验，可选'kendall','spearman'
plt.figure()
plt.title('Spearman Correlation of Features', fontsize = 14)
# colormap = plt.cm.viridis
sns.heatmap(data=dcorr,
            # cmap=colormap,
            linewidths=0.1, vmax=1.0 ,fmt=".2f", square=True, annot=True,annot_kws={'size':14,'weight':'normal', 'color':'white'},mask=np.triu(np.ones_like(dcorr,dtype=np.bool)))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()