#-*-coding:utf-8 -*-
import codecs
import csv
from datetime import datetime
with open('Result_Youtube_Edited_'+ datetime.now().date().strftime('%Y%m%d')+'.csv', 'wb') as csvfile:
    csvfile.write(codecs.BOM_UTF8)
    spamwriter = csv.writer(csvfile, dialect='excel')
    # 读要转换的txt文件，文件每行各词间以@@@字符分隔
    with open('Result_Youtube_Edited.txt', 'rb') as filein:
        for line in filein:
            line_list = line.strip('\n').split('@@@')
            spamwriter.writerow(line_list)