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

cols1 = readxlsx('IPList.xlsx') #这个地方的xlsx要改成你存储游戏名的xlsx，你用sql的话相应的换一下
YsIndex=np.zeros(1200) #这个地方的1200是我表格中的游戏数，有多少游戏就改成多少
for i in range(0, len(cols1),1):
    pytrend.build_payload(kw_list=['Ys', cols1[i]], cat=0, timeframe='2010-01-01 2020-9-7', geo ='', gprop='')
    df = pytrend.interest_over_time()
    print(pytrend.interest_over_time())
    if df['Ys'].sum() == 0:
        YsIndex[i] = df[cols1[i]].sum() / 1
    else:
        YsIndex[i] = df[cols1[i]].sum() / df['Ys'].sum()
    print(YsIndex[i])
    pd.DataFrame(YsIndex).to_csv('IPGoogleTrendsTest.csv') #这个地方输出的csv是我存储游戏名的，你用sql的话相应的换一下