#-*- coding:utf-8 -*-

import re
import time
infp = open('Result_Youtube_Reformatted.txt', "r")
outfp = open('Result_Youtube_Edited.txt', 'w')
lines = infp.readlines()

for i in range(0,len(lines)):
    lines[i]=lines[i].replace('\n','@@@')

# print lines
lines2 = ['视频名称@@@观看次数@@@发布时间@@@播主名称@@@粉丝数量\n']
i=0

list1 = ['VanossGaming', 'jacksepticeye', 'markiplierGAME', 'TheDiamondMinecart', 'theRadBrad', 'MiniLaddd']
dict1 = {'VanossGaming':'2315万', 'jacksepticeye':'1953万', 'markiplierGAME':'2099万', 'TheDiamondMinecart':'1943万', 'MiniLaddd':'482万', 'theRadBrad':'900万'}
for j in range(0, len(lines)-6, 3):
    addthing = lines[j] + lines[j+1] + lines[j+2] + list1[i] + '@@@' + dict1[list1[i]]
    lines2.append(addthing+'\n')
    # print addthing
    if (re.search('周前', lines[j+2]) != None and re.search('周前', lines[j+5]) == None and re.search('月前', lines[j+5]) == None):
        i=i+1
    elif (re.search('月前', lines[j+2]) != None and re.search('月前', lines[j+5]) == None):
        i=i+1

outfp.writelines(lines2[0:500])
