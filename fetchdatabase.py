
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


# 爬虫函数
def crawl(url):
    driver = webdriver.Firefox(executable_path='C:/Python27/geckodriver')
    driver.get(url)
    content = driver.find_elements_by_xpath("//div[@class='tit']")

    for tag in content:
        print tag.text
        infofile.write(tag.text + "\r\n")


        print ' '
    time.sleep(2)
    driver.quit()
    # 主函数


if __name__ == '__main__':
    infofile = codecs.open("Result_Youmin.txt", 'w+', 'utf-8')


    i = 0

    list1 = ['jacksepticeye']
    while i < len(list1):
        url = 'http://ku.gamersky.com/release/pc_201904/'
        print url

        crawl(url)
        infofile.write("\r\n\r\n\r\n")
        i = i + 1

    infofile.close()