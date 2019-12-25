from pytrends.request import TrendReq
from pandas import Series,DataFrame
import pandas as pd
import csv
import xlrd
import time

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()


def readxlsx(filename):
    ExcelFile = xlrd.open_workbook(filename)
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    del cols[0]
    return cols

cols1 = readxlsx('GameList.xlsx')

for i in range(1100, 1110,4):
    pytrend.build_payload(kw_list=['House Flipper', cols1[i], cols1[i+1], cols1[i+2], cols1[i+3]], cat=0, timeframe='2013-01-01 2019-12-25', geo ='', gprop='')
    interest_over_time_df = pytrend.interest_over_time()
    print(pytrend.interest_over_time())
    interest_over_time_df.to_csv('GoogleTrendsTest'+ str(i) +'.csv',header=True,index=True)