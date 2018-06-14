import os
import time
os.system("python ./fetchbilibili.py")
time.sleep(5)
os.system("python ./deleteblanklinebilibili.py")
time.sleep(5)
os.system("python ./3lines21bilibili.py")
time.sleep(5)
os.system("python ./text2csvbilibili.py")

