import os
import time
os.system("python ./fetchtwitch.py")
time.sleep(5)
os.system("python ./twitchtxt2csv.py")

