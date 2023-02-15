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
    filename = r'C:/Python27/Companies Finalized.xls'
    cols = readxlsx(filename)
    company_location_list = []
    company_founded_list = []
    company_parent_list = []
    company_subsid_list = []
    company_key_people_list = []
    company_fte_list = []
    company_type_list = []
    company_owner_list = []

    # 读xlsx表
    for i in cols:
        time.sleep(1)
        try:
            infobox = wptools.page(i).get_parse().data['infobox']
            if infobox.__contains__('hq_location_country'):
                company_location_list.append(infobox['hq_location_country'])
            elif infobox.__contains__('location_country'):
                company_location_list.append(infobox['location_country'])
            elif infobox.__contains__('hq_location'):
                company_location_list.append(infobox['hq_location'])
            elif infobox.__contains__('location'):
                company_location_list.append(infobox['location'])
            elif infobox.__contains__('country'):
                company_location_list.append(infobox['country'])
            elif infobox.__contains__('headquarters'):
                company_location_list.append(infobox['headquarters'])
            elif infobox.__contains__('founded'):
                company_location_list.append(infobox['founded'])
            elif infobox.__contains__('birth_place'):
                company_location_list.append(infobox['birth_place'])
            elif infobox.__contains__('nationality'):
                company_location_list.append(infobox['nationality'])
            else:
                company_location_list.append('Error')

            if infobox.__contains__('founded'):
                company_founded_list.append(infobox['founded'])
            else:
                company_founded_list.append('Error')

            if infobox.__contains__('parent'):
                company_parent_list.append(infobox['parent'])
            else:
                company_parent_list.append('Error')

            if infobox.__contains__('subsid'):
                company_subsid_list.append(infobox['subsid'])
            elif infobox.__contains__('divisions'):
                company_subsid_list.append(infobox['divisions'])
            else:
                company_subsid_list.append('Error')

            if infobox.__contains__('key_people'):
                company_key_people_list.append(infobox['key_people'])
            else:
                company_key_people_list.append('Error')

            if infobox.__contains__('num_employees'):
                company_fte_list.append(infobox['num_employees'])
            else:
                company_fte_list.append('Error')

            if infobox.__contains__('type'):
                company_type_list.append(infobox['type'])
            else:
                company_type_list.append('Error')

            if infobox.__contains__('owner'):
                company_owner_list.append(infobox['owner'])
            else:
                company_owner_list.append('Error')
                
        except:
            company_location_list.append('Error')
            company_founded_list.append('Error')
            company_parent_list.append('Error')
            company_subsid_list.append('Error')
            company_key_people_list.append('Error')
            company_fte_list.append('Error')
            company_type_list.append('Error')
            company_owner_list.append('Error')
            pass

        df0 = [cols, company_location_list, company_founded_list, company_parent_list, company_subsid_list,
               company_key_people_list, company_fte_list, company_type_list, company_owner_list]
        df1 = pd.DataFrame(df0).T
        df1.to_excel("company_info_6.xlsx")
