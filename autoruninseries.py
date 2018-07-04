import os
import time
os.system("python ./youtuberetrieve.py")
time.sleep(3)
os.system("python ./deleteblanklines.py")
time.sleep(2)
os.system("python ./3lines21.py")
time.sleep(2)
os.system("python ./text2csv.py")
time.sleep(2)
os.system("python ./csv2xlsx.py")

