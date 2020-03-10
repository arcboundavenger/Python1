import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv('GamesalesdataV3.csv')
commutes = X.Ratings

commutes.plot.hist(grid=True, bins=10, rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)
plt.show()