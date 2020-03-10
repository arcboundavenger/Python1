import csv
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


data = pd.read_csv('Gamesdataempty.csv')

df = pd.DataFrame(KNN(k=6).fit_transform(data), dtype=int)
df.to_csv("Playwithempty_KNN.csv",index=False,sep=',')

# imputer = KNNImputer(n_neighbors=6)
# df =  pd.DataFrame(imputer.fit_transform(data), dtype=int)
# df.to_csv("Playwithempty_KNN_2.csv",index=False,sep=',')


# df = pd.DataFrame(IterativeImputer(max_iter=10, random_state=0).fit_transform(data), dtype=int)
# df.to_csv("Playwithempty_MICE.csv",index=False,sep=',')

# df = pd.DataFrame(data.interpolate(method = 'linear'), dtype=int)
# df.to_csv("Playwithempty_Interpolation.csv",index=False,sep=',')