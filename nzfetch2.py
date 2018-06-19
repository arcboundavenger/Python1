
# -*- coding: utf-8 -*-
"""
Created on 2018-06-09 23:02

@author: JCMA
"""

import time
import re
import sys
import codecs
import urllib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import codecs
import csv
from datetime import datetime


# 爬虫函数
def crawl(url):
    driver = webdriver.Firefox(executable_path='C:/Python27/geckodriver')
    driver.get(url)
    content = driver.find_elements_by_xpath("//h2[@class='entry-title']")

    for tag in content:
        print tag.text
        infofile.write(tag.text + "\r\n")
        print ' '
    time.sleep(5)
    driver.quit()
    # 主函数


if __name__ == '__main__':
    infofile = codecs.open("Result_NZ.txt", 'w+', 'utf-8')
    i = 0
    url='http://nzgda.com/category/games/'
    crawl(url)
    # list1 = ['jacksepticeye', 'markiplierGAME']
    # #list1 = ['jacksepticeye']
    # while i < len(list1):
    #     url = 'https://www.youtube.com/user/' + str(list1[i]) + '/videos'
    #     print url
    #
    #     crawl(url)
    #     infofile.write("\r\n\r\n\r\n")
    #     i = i + 1

    infofile.close()
    with open('Result_NZ.csv', 'wb') as csvfile:
        csvfile.write(codecs.BOM_UTF8)
        spamwriter = csv.writer(csvfile, dialect='excel')
        # 读要转换的txt文件，文件每行各词间以@@@字符分隔
        with open('Result_NZ.txt', 'rb') as filein:
            for line in filein:
                line_list = line.strip('\n').split('@@@')
                spamwriter.writerow(line_list)