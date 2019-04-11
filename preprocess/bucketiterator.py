# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import torch
from collections import Counter
import pandas as pd

import random
import pickle
import preprocess
from tools.timer import log_time_delta

from nltk.corpus import stopwords
Overlap = 237
import random
from units import to_array, pad_sequence
from tools import evaluation

class BucketIterator(object):
    def __init__(self,data,opt=None,batch_size=2,batch_num = 0, max_sequence_length=0,shuffle=True,test=False,position=False,backend="keras",need_balanced = False,always = False,balance_temperature=1):
        self.shuffle=shuffle
        self.data=data
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.test=test 
        self.backend=backend
        self.transform=self.setTransform()
        self.always=always
        self.max_sequence_length = max_sequence_length
        self.need_balanced = need_balanced
        self.balance_temperature=balance_temperature

        
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
        
        padded_sequences = [pad_sequence(s, maxlen = self.max_sequence_length) for s in data[0]]
        x_data = torch.Tensor(padded_sequences)
        y_data = torch.Tensor(data[1])
        return {'X':x_data, 'y':y_data}
    
#    def transformTorch(self,data):
#        if torch.cuda.is_available():
#            data=data.reset_index()
#            text= Variable(torch.LongTensor(data.text).cuda())
#            label= Variable(torch.LongTensor([int(i) for i in data.label.tolist()]).cuda())                
#        else:
#            data=data.reset_index()
#            text= Variable(torch.LongTensor(data.text))
#            label= Variable(torch.LongTensor(data.label.tolist()))
#        if self.position:
#            position_tensor = self.get_position(data.text)
#            return DottableDict({"text":(text,position_tensor),"label":label})
#        return DottableDict({"text":text,"label":label})
    
    
    def transformKeras(self,data):
        list_of_data = []
        for i in data:
            if type(i[0])!=int and type(i)!=np.ndarray and type(i[0][0])==int:
                list_of_data.append(to_array(i,self.max_sequence_length, use_mask = False))
            else:
                list_of_data.append(np.asarray(i))
                    
        return list_of_data
#        return [to_array(i,self.max_sequence_length, use_mask = False) if type(i[0])!=int and type(i)!=np.ndarray  else i for i in data]
    
    def transformTF(self,data):
        
        return [to_array(i,self.max_sequence_length) if type(i[0])!=int and type(i)!=np.ndarray  else i for i in data]
    
    def balance(self,data):
        
        lenght_per_label=sum(data[-1])
        
        from functools import reduce
        product = reduce((lambda x, y: x * y), lenght_per_label,1)
        
        importance = product/lenght_per_label
        importance= importance /sum(importance)
        exp = np.exp(importance * self.balance_temperature ) 
#        exp = exp /np.sum(exp)
#        ratio =  product/lenght_per_label
        ratio = exp/max(exp) 
        data_groupby_label = dict()
        for i in range(len(data[0])):
            label = data[-1][i].argmax()
            data_groupby_label.setdefault(label,[])
            data_groupby_label[label].append(i)
        
        balance_index = []
        for j in range(data[-1].shape[-1]):
            k = int(len(data_groupby_label[j]) * ratio[j])
            balance_index.extend( random.sample(data_groupby_label[j],k))
        
        index_by_list = lambda L, Idx :  [L[i] for i in Idx]
        
        return [index_by_list(feature,balance_index) if type(feature) == list  else feature[balance_index] for feature in data]
        
    def __iter__each(self,data):
        if (self.shuffle and not self.test) or self.need_balanced:
            c = list(zip(*data))
            random.shuffle(c)
            data = [i for i in zip(*c)]
        
        if self.batch_num == 0:
           
            batch_num = int(len(data[1])/self.batch_size)
        else:
            batch_num = self.batch_num

       
        indexes = [(i*self.batch_size,(i+1)*self.batch_size) for i in range(batch_num)]
        if len(c)%self.batch_size!=0:
           indexes.append((len(c)-self.batch_size,len(c)))

        for index in indexes:
#            yield self.transform([item[index[0]:index[1]] for item in self.data])
            yield self.transform([item[index[0]:index[1]] for item in data])
        

    def __iter__(self):
        
        if self.need_balanced:
            self.data=self.balance(self.data)
            
        if not self.always:
            for sample in self.__iter__each(self.data):
    #            yield self.transform([item[index[0]:index[1]] for item in self.data])
                yield sample
        else:
            while True:
                for sample in self.__iter__each(self.data):
    #            yield self.transform([item[index[0]:index[1]] for item in self.data])
                    yield sample