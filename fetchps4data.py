import time
from selenium import webdriver
import xlrd

def readxlsx(filename):
    ExcelFile = xlrd.open_workbook(filename)
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    del cols[0]
    return cols

if __name__ == '__main__':
    filename = r'C:/Python27/sheet1.xlsx'
    cols = readxlsx(filename)
    newcols = []
    # 读xlsx表

    driver = webdriver.Firefox(executable_path='C:/Python27/geckodriver')
    url = 'http://gamstat.com/games'
    driver.set_page_load_timeout(600)
    # 网络不太好，避免timeout把时间设成10分钟

    try:
        driver.get(url)
        time.sleep(2)
        # 打开网页
        js2 = "var q=document.getElementById('label_ps4').click()"
        driver.execute_script(js2)
        time.sleep(2)
        # 点击PS4字样

        for i in cols:
            textarea = driver.find_elements_by_xpath("//div[@id='games_filter']//input[@type='search']")[0]
            # 找到想要的文字输入框
            textarea.send_keys(' ' + i)
            time.sleep(5)
            # 输入文字

            try:
                content = driver.find_element_by_xpath('//tbody/tr[1]/td[5]').text
                print (content)
                newcols.append(content)
                # 爬取表中owner的部分
            except:
                pass
                content = 'N/A'
                newcols.append(content)
                print (content)
                # 如果找不到则打N/A

            textarea.clear()
            time.sleep(5)
            # 删除文字

    except:
        print('timeout')
        pass

