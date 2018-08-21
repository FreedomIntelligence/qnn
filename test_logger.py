# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:14:47 2018

@author: quartz
"""

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='E:/qiuchi/example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
