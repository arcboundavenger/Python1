# -*- coding: utf-8 -*-
import re
def clearBlankLine(infile,outfile):
    file1 = open(infile, 'r') # 要去掉空行的文件
    file2 = open(outfile, 'w') # 生成没有空行的文件
    try:
        line = file1.readlines()
        for i in range(0, len(line)):
            result1 = re.search('\d:\d', line[i])
            if result1 != None:
                line[i] = '\n'
            # result2 = re.search('\d-\d', line[i])
            # if result2 != None:
            #     line[i] = line[i].split()

        for i in range(0,len(line)):
            if line.count('\n')>0:
                line.remove("\n")
            elif line.count('字幕\n') > 0:
                line.remove('字幕\n')

        file2.writelines(line)
    finally:
        file1.close()
        file2.close()


if __name__ == "__main__":
    clearBlankLine("Result_Bilibili.txt","Result_Bilibili_Reformatted.txt")