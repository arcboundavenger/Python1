#!/usr/bin/env python
#!coding=utf-8
from datetime import datetime
datetext = open(datetime.now().date().strftime('%Y%m%d')+'.txt', 'w')
datetext.write('cool')
datetext.close()
