from pytrends.request import TrendReq
import pandas as pd
import xlrd
import numpy as np
import time
import random

pytrend = TrendReq()

df1 = pd.read_csv('GameList.csv') #这个地方的xlsx要改成你存储游戏名的xlsx，你用sql的话相应的换一下
cols1 = df1['Game Name'].to_numpy()
KnackIndex=[None]*(len(cols1)+1) #这个地方的1200是我表格中的游戏数，有多少游戏就改成多少
for i in range(0, len(cols1),1):
    pytrend.build_payload(kw_list=['Knack', cols1[i]], cat=0, timeframe='2013-01-01 2023-7-18', geo ='', gprop='')
    df = pytrend.interest_over_time()
    print(pytrend.interest_over_time())
    KnackIndex[i] = df[cols1[i]].max()/df['Knack'].max()
    print(KnackIndex)
    pd.DataFrame(KnackIndex).to_csv('GoogleTrendsTest.csv')  # 这个地方输出的csv是我存储游戏名的，你用sql的话相应的换一下
    time.sleep(random.randint(20,30))