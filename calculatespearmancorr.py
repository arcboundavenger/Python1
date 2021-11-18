
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import palettable


# train = pd.read_csv('forspearman.csv')
train = pd.read_csv('forpearson.csv')

dcorr = train.corr(method='spearman')#默认为'pearson'检验，可选'kendall','spearman'
print(dcorr)
dcorr.to_csv('test111.csv') #保存结果
plt.figure()
plt.title('Spearman Correlation of Features', fontsize = 12)

extreme_1 = 0.4  # show with a star
extreme_2 = 0.7  # show with a second star
extreme_3 = 0.9  # show with a third star
annot = [[f"{val:.2f}"
          + ('' if abs(val) < extreme_1 else '\n★')  # add one star if abs(val) >= extreme_1
          + ('' if abs(val) < extreme_2 else '★')  # add an extra star if abs(val) >= extreme_2
          + ('' if abs(val) < extreme_3 else '★')  # add yet an extra star if abs(val) >= extreme_3
          for val in row] for row in dcorr.to_numpy()]
sns.heatmap(data=dcorr,
            cmap=plt.cm.viridis_r,
            linewidths=0.1, vmin=-1, vmax=1.0 ,fmt="", square=True, annot=annot,annot_kws={'size':9,'weight':'bold', 'color':'white'},mask=np.triu(np.ones_like(dcorr,dtype=np.bool)),
            xticklabels=True, yticklabels=True)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.show()