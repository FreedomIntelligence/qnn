# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

from collections import Counter
import pandas as pd

import random
import pickle
import preprocess
from tools.timer import log_time_delta

from nltk.corpus import stopwords
Overlap = 237
import random
from units import to_array 
from tools import evaluation

class BucketIterator(object):
    def __init__(self,data,always=False,opt=None,batch_size=2,max_sequence_length=0,shuffle=True,test=False,position=False,backend="keras"):
        self.shuffle=shuffle
        self.data=data
        self.batch_size=batch_size
        self.test=test 
        self.backend=backend
        self.transform=self.setTransform()
        self.always=always
        self.max_sequence_length = max_sequence_length
        
        if opt is not None:
            self.setup(opt)
            
    def setup(self,opt):
        
        self.batch_size=opt.batch_size
        self.shuffle=opt.__dict__.get("shuffle",self.shuffle)
#        self.position=opt.__dict__.get("position",False)
        self.transform=self.setTransform()
        
    def setTransform(self):
        if self.backend == "tensorflow":
            return self.transformTF
        elif self.backend == "torch":
            return  self.transformTorch
        else:
            return  self.transformKeras
            
    
    def transformTorch(self,data):
        if torch.cuda.is_available():
            data=data.reset_index()
            text= Variable(torch.LongTensor(data.text).cuda())
            label= Variable(torch.LongTensor([int(i) for i in data.label.tolist()]).cuda())                
        else:
            data=data.reset_index()
            text= Variable(torch.LongTensor(data.text))
            label= Variable(torch.LongTensor(data.label.tolist()))
        if self.position:
            position_tensor = self.get_position(data.text)
            return DottableDict({"text":(text,position_tensor),"label":label})
        return DottableDict({"text":text,"label":label})
    
    
    def transformKeras(self,data):
        return [to_array(i,self.max_sequence_length, use_mask = False) if type(i[0])!=int and type(i)!=np.ndarray  else i for i in data]
    
    def transformTF(self,data):
        
        return [to_array(i,self.max_sequence_length) if type(i[0])!=int and type(i)!=np.ndarray  else i for i in data]
    
    def __iter__(self):
        if self.shuffle and not self.test:
            c = list(zip(*self.data))
            random.shuffle(c)
            self.data = [i for i in zip(*c)]

            
        batch_nums = int(len(self.data[0])/self.batch_size)

       
        indexes = [(i*self.batch_size,(i+1)*self.batch_size) for i in range(batch_nums)]
        if len(self.data)%self.batch_size!=0:
           indexes.append((len(self.data[0])-self.batch_size,len(self.data[0])))

        for index in indexes:

            yield self.transform([item[index[0]:index[1]] for item in self.data])