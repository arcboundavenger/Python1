#-*- coding:utf-8 -*-
import re


infp = open('Result_Bilibili_Reformatted.txt', "r")
outfp = open('Result_Bilibili_Edited.txt', 'w')
lines = infp.readlines()

for i in range(0,len(lines)):

    lines[i]=lines[i].replace('\n','@@@')
    result2 = re.search('\d-\d', lines[i])
    if result2 != None:
        lines[i]=lines[i].replace(' ','@@@')

print lines
lines2 = []
i=0
for j in range(0, len(lines)-2, 2):
    addthing = lines[j] + lines[j+1]
    lines2.append(addthing+'\n')


outfp.writelines(lines2)