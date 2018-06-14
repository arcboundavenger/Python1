#-*-coding:utf-8 -*-
import codecs
import csv
from datetime import datetime
with open('Result_Twitch_Edited_'+ datetime.now().date().strftime('%Y%m%d')+'.csv', 'wb') as csvfile:
    csvfile.write(codecs.BOM_UTF8)
    spamwriter = csv.writer(csvfile, dialect='excel')
    with open('Result_Twitch.txt', 'rb') as filein:
        lines = []
        for line in filein:
            lines.append(line)
        for i in range(0, len(lines)-2, 2):
            spamwriter.writerow([lines[i],lines[i+1]])
