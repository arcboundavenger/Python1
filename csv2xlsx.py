#-*-coding:utf-8 -*-
import pandas as pd
from datetime import datetime
import openpyxl
def csv_to_xlsx_pd():
    csv = pd.read_csv('Result_Youtube_Edited_'+ datetime.now().date().strftime('%Y%m%d')+'.csv', encoding='utf-8')
    csv.to_excel('Result_Youtube_Edited_'+ datetime.now().date().strftime('%Y%m%d')+'.xlsx', sheet_name='data')


if __name__ == '__main__':
    csv_to_xlsx_pd()