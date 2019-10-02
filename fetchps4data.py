import time
from selenium import webdriver
import xlrd
import xlsxwriter

if __name__ == '__main__':

    ExcelFile = xlrd.open_workbook(r'C:/Python27/sheet1.xlsx')
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    del cols[0]
    newcols = []
    #读xlsx表

    workbook = xlsxwriter.Workbook('result_ps4_Exclusive.xlsx')
    worksheet = workbook.add_worksheet(u'sheet1')
    #生成空表

    driver = webdriver.Firefox(executable_path='C:/Python27/geckodriver')
    url = 'http://gamstat.com/games'
    driver.get(url)
    #打开网页

    for i in cols:
        time.sleep(5)
        js2 = "var q=document.getElementById('label_ps4').click()"
        driver.execute_script(js2)
        #点击PS4字样
        time.sleep(2)
        textarea = driver.find_elements_by_xpath("//div[@id='games_filter']//input[@type='search']")[0]
        #找到想要的文字输入框
        textarea.send_keys(i)
        time.sleep(5)
        try:
            content = driver.find_element_by_xpath('//tbody/tr[1]/td[5]').text
            print (content)
            newcols.append(content)
        except:
            pass
            content = 'N/A'
            newcols.append(content)
            print (content)
        #一个有可能遇到错误的循环
        driver.refresh()
        #存储爬取的数据


    worksheet.write_column('A1', newcols)
    workbook.close()
    #写入新表
