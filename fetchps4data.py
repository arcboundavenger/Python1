import time
import re
import sys
import codecs
import urllib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pymouse,pykeyboard,os,sys
from pymouse import *
from pykeyboard import PyKeyboard
import xlrd

# 爬虫函数
def crawl(url):
    driver = webdriver.Firefox(executable_path='C:/Python27/geckodriver')
    driver.get(url)
    textarea = driver.find_elements_by_xpath("//input[@type='search']")
    #content = driver.find_elements_by_xpath("//div[@class='content clearfix']/div[@class='small-item fakeDanmu-item']")
    #content = driver.find_elements_by_xpath("//div[@class='h-basic-spacing']")
    print (content)
    # infofile.write(url + "\r\n")
    js2 = "var q=document.getElementById('label_ps4').click()"
    driver.execute_script(js2)
    textarea = driver.find_elements_by_xpath("//div[@id='games_filter']//input[@type='search']")[0]
    textarea.send_keys('asdkjhf')
    for tag in content:
        #print (tag.text)
        infofile.write(tag.text + "\r\n")
        print (' ')
    time.sleep(5)
    driver.quit()
    # 主函数


if __name__ == '__main__':
    ExcelFile = xlrd.open_workbook(r'C:/Python27/sheet1.xlsx')
    sheet = ExcelFile.sheet_by_index(0)
    cols = sheet.col_values(0)
    dels cols[0]

    infofile = codecs.open("Result_PS4.txt", 'w', 'utf-8')

    i = 0
    url=''

    list1 = [2728123]
    while i < len(list1):
        url = 'https://gamstat.com/games/'
        print (url)
        crawl(url)
        infofile.write("\r\n\r\n\r\n")
        i = i + 1

    infofile.close()

