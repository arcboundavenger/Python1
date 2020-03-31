from pytrends.request import TrendReq
from pandas import Series,DataFrame
import pandas as pd
import csv
import xlrd
import time
import numpy as np

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()


def readxlsx(filename):
    ExcelFile = xlrd.open_workbook(filename)
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    del cols[0]
    return cols

cols1 = readxlsx('GameList.xlsx')
KnackIndex=np.zeros(1200)
for i in range(0, len(cols1),1):
    pytrend.build_payload(kw_list=['Knack', cols1[i]], cat=0, timeframe='2013-01-01 2020-3-20', geo ='', gprop='')
    df = pytrend.interest_over_time()
    print(pytrend.interest_over_time())
    # df.to_csv('GoogleTrendsTest'+ str(i) +'.csv',header=True,index=True)
    KnackIndex[i] = df[cols1[i]].max()/df['Knack'].max()
    print(KnackIndex[i])
    # KnackIndex[i+1] = df[cols1[i+1]].argmax()/df['Knack'].argmax()
    # KnackIndex[i+2] = df[cols1[i+2]].argmax()/df['Knack'].argmax()
    # KnackIndex[i+3] = df[cols1[i+3]].argmax()/df['Knack'].argmax()
    pd.DataFrame(KnackIndex).to_csv('GoogleTrendsTest.csv')

