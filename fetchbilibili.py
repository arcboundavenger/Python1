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
    #content = driver.find_elements_by_xpath("//div[@class='content clearfix']/div[@class='small-item fakeDanmu-item']")
    content = driver.find_elements_by_xpath("//div[@class='h-basic-spacing']")
    print (content)
    # infofile.write(url + "\r\n")

    for tag in content:
        #print (tag.text)
        infofile.write(tag.text + "\r\n")
        print (' ')
    time.sleep(5)
    driver.quit()
    # 主函数


if __name__ == '__main__':
    infofile = codecs.open("Result_Bilibili.txt", 'w', 'utf-8')

    i = 0
    url=''

    # list1 = [168598, 423895, 2728123, 2771237, 2058048, 70666, 13308108, 419220, 433351, 24754667, 562197, 2019740]
    list1 = [2728123]
    while i < len(list1):
        url = 'https://space.bilibili.com/' + str(list1[i]) + '/#/'
        print (url)

        crawl(url)
        infofile.write("\r\n\r\n\r\n")
        i = i + 1

    infofile.close()

