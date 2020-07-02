import datetime

gotDay = datetime.date(2020,1,1)
epoch = datetime.date(1900,1,1)

print((gotDay-epoch).days)