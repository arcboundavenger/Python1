import re
line = ['1:2', '2:34', '123', '192', 'OI:D', 'IJ:2']
for i in range(0, len(line)):
    result1 = re.search('\d:\d', line[i])
    if result1 != None:
        line[i] = '\n'

print line