import pandas as pd
import jenkspy
import matplotlib.pyplot as plt
import numpy as np

df2 = pd.read_csv('test1ff7core.csv')
df2['ds'] = pd.to_datetime(df2['ds'])
df2.set_index(df2['ds'], inplace = True)
ts = df2['y']
y = np.array(ts.tolist())
n_breaks = 10
breaks = jenkspy.jenks_breaks(y, n_breaks-1)
breaks_jkp = []
breaks_jkp_str = []
for v in breaks:
    idx = ts.index[ts == v]
    breaks_jkp.append(idx)
    breaks_jkp_str.append(idx.strftime('%Y-%m-%d')[0])
print(breaks_jkp_str)

plt.plot(ts)
print_legend = True
for i in breaks_jkp:
    if print_legend:
        plt.axvline(i, color='red',linestyle='dashed', label='breaks')
        print_legend = False
    else:
        plt.axvline(i, color='red',linestyle='dashed')
plt.grid()
plt.legend()
plt.show()