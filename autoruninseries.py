import os
import time
os.system("python ./youtuberetrieve.py")
time.sleep(5)
os.system("python ./deleteblanklines.py")
time.sleep(5)
os.system("python ./3lines21.py")
time.sleep(5)
os.system("python ./text2csv.py")


