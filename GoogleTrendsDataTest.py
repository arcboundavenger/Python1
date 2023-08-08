from pytrends.request import TrendReq
import pandas as pd
import xlrd
import numpy as np
import time
import random
from pytrends.exceptions import ResponseError




df1 = pd.read_csv('GameList.csv') #读取csv
cols1 = df1['Game Name'].to_numpy()
KnackIndex=[None]*(len(cols1)+1)


for i in range(0, len(cols1), 1):
    while True:
        try:
            # 在此处编写您的代码
            pytrend = TrendReq(timeout=(10,25))
            pytrend.build_payload(kw_list=['Knack', cols1[i]], cat=0, timeframe='2013-01-01 2023-7-18', geo='',
                                  gprop='')
            df = pytrend.interest_over_time()
            print(pytrend.interest_over_time())
            KnackIndex[i] = df[cols1[i]].max() / df['Knack'].max()
            print(KnackIndex)
            pd.DataFrame(KnackIndex).to_csv('GoogleTrendsTest.csv')  # 这个地方输出的csv是我存储游戏名的，你用sql的话相应的换一下
            break  # 如果代码成功运行，则跳出循环
        except ResponseError as e:
            print('请求过多错误：', e)
            print('等待60秒后重试...')
            time.sleep(60)  # 等待60秒后重试




