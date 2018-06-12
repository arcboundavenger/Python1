# -*- coding: utf-8 -*-
import re
def clearBlankLine():
    file1 = open('Result_Youtube.txt', 'r') # 要去掉空行的文件
    file2 = open('Result_Youtube_Reformatted.txt', 'w') # 生成没有空行的文件
    try:
        line = file1.readlines()
        for i in range(0, len(line)):
            result1 = re.search('\d:\d', line[i])
            if result1 != None:
                line[i] = '\n'
        for i in range(0,len(line)):
            if line.count('\n')>0:
                line.remove("\n")
            elif line.count('字幕\n') > 0:
                line.remove('字幕\n')

        file2.writelines(line)
    finally:
        file1.close()
        file2.close()


if __name__ == '__main__':
    clearBlankLine()