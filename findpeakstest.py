
import peakutils
from peakutils.plot import plot as pplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


milk_data = pd.read_csv('peaktest111.csv')
time_series = milk_data['y']
y = np.asarray(time_series)

df = milk_data[0:]

x = milk_data['ds']


indexes = peakutils.indexes(y, thres=0.3, min_dist=1)

pplot(x, y, indexes)

baseline_values = peakutils.baseline(time_series, deg=0)

plt.plot(x, baseline_values)
plt.show()
print(baseline_values)

# plt.plot(x, y2)
# plt.show()
# base = peakutils.baseline(y2, 2)
