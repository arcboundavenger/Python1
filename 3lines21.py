#-*- coding:utf-8 -*-

import re

infp = open('Result_Youtube_Reformatted.txt', "r")
outfp = open('Result_Youtube_Edited.txt', 'w')
lines = infp.readlines()

for i in range(0,len(lines)):
    lines[i]=lines[i].replace('\n','@@@')

print lines
lines2 = []
i=0
for j in range(0, len(lines)-3, 3):
    addthing = lines[j] + lines[j+1] + lines[j+2]
    lines2.append(addthing+'\n')


outfp.writelines(lines2)
