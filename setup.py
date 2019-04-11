# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 10:08:34 2018

@author: qiuchi
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
      name = 'qnn',
      version = '1.0.0',
      install_requires=[
              'numpy>=1.8.2',
              'scipy>=0.13.3',
              'nltk>=3.3',
              'tensorflow>=1.12.0',
              'keras>=2.2.4',
              'torch>=1.0.0',
              'torchvision>=0.2.1',
              'sklearn',
              'keras-bert',
              'jieba>=0.39',
              'gensim>=3.6.0']
      )

import nltk
nltk.download('punkt')
nltk.download('stopwords')
