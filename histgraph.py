import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv('GamesalesdataV3.csv')
commutes = X.MCUserRatings

commutes.plot.hist(grid=True, bins=100, rwidth=0.9,
                   color='#607c8e')


plt.xlabel('Ratings')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.show()