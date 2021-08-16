import wptools
import xlrd
import time
import pandas as pd


def readxlsx(filename):
    ExcelFile = xlrd.open_workbook(filename)
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    del cols[0]
    return cols
if __name__ == '__main__':
    filename = r'C:/Python27/sheet1.xls'
    cols = readxlsx(filename)
    company_location_list = []
    # 读xlsx表
    for i in cols:
        # time.sleep(1)
        try:
            so = wptools.page(i).get_parse()
            infobox = so.data['infobox']
            if infobox.__contains__('num_employees'):
                company_location_list.append(infobox['num_employees'])
            else:
                company_location_list.append('Error')
        except:
            company_location_list.append('Error')
            pass

    df1 = pd.DataFrame(company_location_list, columns=['FTEs'])
    df1.to_excel("company_Ftes.xlsx")