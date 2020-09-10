from scipy import stats
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

dat = pd.read_csv('anova_test.csv')
dat.head()
formula = 'Group~C(Developer)+C(Publisher)+C(IP)+C(PlayerType)'
model = ols(formula,dat).fit()
anovat = anova_lm(model, typ=2)
print(anovat)

