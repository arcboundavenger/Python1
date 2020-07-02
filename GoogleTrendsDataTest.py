from pytrends.request import TrendReq
import pandas as pd
import xlrd
import numpy as np

pytrend = TrendReq()


def readxlsx(filename):
    ExcelFile = xlrd.open_workbook(filename)
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    del cols[0]
    return cols

cols1 = readxlsx('GameList.xlsx') #这个地方的xlsx要改成你存储游戏名的xlsx，你用sql的话相应的换一下
KnackIndex=np.zeros(1200) #这个地方的1200是我表格中的游戏数，有多少游戏就改成多少
for i in range(0, len(cols1),1):
    pytrend.build_payload(kw_list=['Knack', cols1[i]], cat=0, timeframe='2013-01-01 2020-3-20', geo ='', gprop='')
    df = pytrend.interest_over_time()
    print(pytrend.interest_over_time())
    KnackIndex[i] = df[cols1[i]].max()/df['Knack'].max()
    print(KnackIndex[i])
    pd.DataFrame(KnackIndex).to_csv('GoogleTrendsTest.csv') #这个地方输出的csv是我存储游戏名的，你用sql的话相应的换一下