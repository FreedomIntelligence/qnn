#-*- coding:utf-8 -*-

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
from preprocess.dictionary import Dictionary
from preprocess.embedding import Embedding
from preprocess.bucketiterator import BucketIterator
from keras.utils import to_categorical


class DataReader(object):
    def __init__(self,opt):
        self.onehot = True
        self.unbalanced_sampling = False
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)        
      
        self.dir_path = os.path.join(opt.datasets_dir, 'QA', opt.dataset_name.lower())
        self.preprocessor = preprocess.setup(opt)
        self.datas = self.load(do_filter = opt.remove_unanswered_question)
        self.get_max_sentence_length()
        self.nb_classes = 2
        self.dict_path = os.path.join(self.bert_dir,'vocab.txt')
        
        if bool(self.bert_enabled):
            loaded_dic = Dictionary(dict_path =self.dict_path)
            self.embedding = Embedding(loaded_dic,self.max_sequence_length)
        else:
            self.embedding = Embedding(self.get_dictionary(self.datas.values()),self.max_sequence_length)
            
        self.embedding = Embedding(self.get_dictionary(self.datas.values()),self.max_sequence_length)

#        self.q_max_sent_length = q_max_sent_length
#        self.a_max_sent_length = a_max_sent_length

        print('loading word embedding...')
        if opt.dataset_name=="NLPCC":     # can be updated
            self.embedding.get_embedding(dataset_name = self.dataset_name, language="cn",fname=opt.wordvec_path) 
        else:
            self.embedding.get_embedding(dataset_name = self.dataset_name, fname=opt.wordvec_path)
        self.opt_callback(opt) 
        
       
        
        
#        opt.embedding_size = self.embeddings.shape[1]

#        self.optCallback(opt)            
    def opt_callback(self,opt):
        opt.nb_classes = self.nb_classes            
        opt.embedding_size = self.embedding.lookup_table.shape[1]        
        opt.max_sequence_length= self.max_sequence_length
        
        opt.lookup_table = self.embedding.lookup_table 
            
    def load(self, do_filter = True):
        datas = dict()
        clean_set = ['test','dev'] if self.train_verbose else ['train','test','dev']
        for data_name in ['train','test']: #'dev'            
            data_file = os.path.join(self.dir_path,data_name+".txt")
            data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"]).fillna('0')
            if do_filter == True and data_name in clean_set:
                data=self.remove_unanswered_questions(data)
                
            data['question'] = data['question'].apply(lambda x : self.preprocessor.run(x,output_type = 'string'))
            data['answer'] = data['answer'].apply(lambda x : self.preprocessor.run(x,output_type = 'string'))
            datas[data_name] = data
        return datas
    
    
    
    @log_time_delta
    def remove_unanswered_questions(self,df):
        counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
        questions_have_correct = counter[counter>0].index
#        counter= df.groupby("question").apply(lambda group: sum(group["flag"]==0))
#        questions_have_uncorrect=counter[counter>0].index
#        counter=df.groupby("question").apply(lambda group: len(group["flag"]))
#        questions_multi=counter[counter>1].index
    
        return df[df["question"].isin(questions_have_correct) ].reset_index()  #&  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)
    
    def get_max_sentence_length(self):
        q_max_sent_length = max(map(lambda x:len(x),self.datas["train"]['question'].str.split()))
        a_max_sent_length = max(map(lambda x:len(x),self.datas["train"]['answer'].str.split()))    
        self.max_sequence_length = max(q_max_sent_length,a_max_sent_length)
        if self.max_sequence_length > self.max_len:
            self.max_sequence_length = self.max_len
                
    def get_dictionary(self,corpuses = None,dataset="",fresh=True):
        pkl_name="temp/"+self.dataset_name+".alphabet.pkl"
        if os.path.exists(pkl_name) and not fresh:
            return pickle.load(open(pkl_name,"rb"))
        dictionary = Dictionary(start_feature_id = 0)
        dictionary.add('[UNK]')  
#        alphabet.add('END') 
        for corpus in corpuses:
            for texts in [corpus["question"].unique(),corpus["answer"]]:    
                for sentence in texts:                   
                    tokens = sentence.lower().split()
                    for token in set(tokens):
                        dictionary.add(token)
        print("alphabet size = {}".format(len(dictionary.keys())))
        if not os.path.exists("temp"):
            os.mkdir("temp")
        pickle.dump(dictionary,open(pkl_name,"wb"))
        return dictionary   
    
    def get_train_2(self,shuffle = True,iterable=True, max_sequence_length=0,overlap_feature = False,sampling_per_question = False,need_balanced=False,always = False,balance_temperature=1):
        
        x_data = []
        num_samples = 0
        if sampling_per_question: 
            #sampling on a per-question basis           
            q = []
            pos_a = []
            neg_a = []
            a = []
            overlap_pos = []
            overlap_neg = []
            y = []
            for question,group in self.datas["train"].groupby("question"):
                seq_q = self.embedding.text_to_sequence(question)
                pos_answers = group[group["flag"] == 1]["answer"]
                neg_answers = group[group["flag"] == 0]["answer"]#.reset_index()
                if len(pos_answers)==0 or len(neg_answers)==0:
                    continue
                
                for pos in pos_answers:  
                    
                    seq_pos_a = self.embedding.text_to_sequence(pos)
                    neg_index = np.random.choice(neg_answers.index)
                    neg = neg_answers.loc[neg_index,]
                    seq_neg_a = self.embedding.text_to_sequence(neg)
                    if self.match_type == 'pointwise': 
                        q = q+[seq_q,seq_q]
                        a = a+[seq_pos_a,seq_neg_a]
                        y = y+[1,0]
                        num_samples = num_samples + 2
                    else:
                        q.append(seq_q)
                        num_samples = num_samples + 1
                        
                    pos_a.append(seq_pos_a)
                    neg_a.append(seq_neg_a)
                    if overlap_feature:
                        overlap_pos.append(self.overlap_index(seq_q,seq_pos_a))
                        overlap_neg.append(self.overlap_index(seq_q,seq_neg_a))
            
            
            if self.bert_enabled:
                q,q_mask = to_array(q,maxlen = self.max_sequence_length, use_mask = True)
                if self.match_type == 'pairwise':      
                    pos_a,pos_a_mask = to_array(pos_a,maxlen = self.max_sequence_length, use_mask = True)
                    neg_a,neg_a_mask = to_array(neg_a,maxlen = self.max_sequence_length, use_mask = True)
                    x_data = [q,q_mask,pos_a,pos_a_mask,neg_a,neg_a_mask]
                    y = [l for l in zip(*[q,pos_a,neg_a])]
                else:
                    y = to_categorical(np.asarray(y))
                    a,a_mask = to_array(a,maxlen = self.max_sequence_length, use_mask = True)
                    x_data = [q,q_mask,a,a_mask]
            else:
                q = to_array(q,maxlen = self.max_sequence_length, use_mask = False)
                if self.match_type == 'pairwise':
                    pos_a = to_array(pos_a,maxlen = self.max_sequence_length, use_mask = False)
                    neg_a = to_array(neg_a,maxlen = self.max_sequence_length, use_mask = False)
                    x_data = [q,pos_a, neg_a]
                    y = [l for l in zip(*[q,pos_a,neg_a])]
                    
                else:
                    y = to_categorical(np.asarray(y))
                    a = to_array(a,maxlen = self.max_sequence_length, use_mask = False)
                    x_data = [q,a]
            if overlap_feature:
                x_data = x_data + [overlap_pos,overlap_neg]
                
        else:         
            num_samples = int(len(self.datas["train"]))
            #sample on the whole data, only support pointwise match type: x=[q,pos_a],y
            assert self.match_type == 'pointwise'
            
            q = self.datas["train"]["question"]
            a = self.datas["train"]["answer"]
            y = self.datas["train"]["flag"]
            q = [self.embedding.text_to_sequence(sent) for sent in q]
    #        q = to_array(q,maxlen = self.max_sequence_length, use_mask = False)
            a = [self.embedding.text_to_sequence(sent) for sent in a]
    #        a = to_array(a,maxlen = self.max_sequence_length, use_mask = False) 
            y = to_categorical(np.asarray(y))
            
            
            if max_sequence_length == 0:
                max_sequence_length = self.max_sequence_length
            if self.bert_enabled:
                q,q_mask = to_array(q,maxlen = self.max_sequence_length, use_mask = True)
                a,a_mask = to_array(a,maxlen = self.max_sequence_length, use_mask = True)
                x_data = [q,q_mask,a,a_mask]
            else:
                q = to_array(q,maxlen = self.max_sequence_length, use_mask = False)
                a = to_array(a,maxlen = self.max_sequence_length, use_mask = False)
                x_data = [q,a]
                
            if overlap_feature:
                overlap = [self.overlap_index(q_seq,a_seq) for q_seq, a_seq in zip(*x_data)]
                x_data = x_data+overlap
        
        self.num_samples = num_samples
       
        if iterable:
            x = [l for l in zip(*x_data)]
            data = (x,y)
            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,need_balanced=need_balanced,always=always,balance_temperature=balance_temperature).__iter__()
#            return BucketIterator(data,batch_size=self.batch_size, batch_num = int(self.num_samples/self.batch_size),shuffle=True,need_balanced=need_balanced,always=always).__iter__()
        else: 
            return x_data,y
        
    
    
    def get_test_2(self,shuffle = True,iterable=True, max_sequence_length=0,overlap_feature = False):
        
        x_data = []
        #sample on the whole data, only support pointwise match type: x=[q,pos_a],y
        
        q = self.datas["test"]["question"]
        a = self.datas["test"]["answer"]
        y = self.datas["test"]["flag"]
        
        if max_sequence_length == 0:
            max_sequence_length = self.max_sequence_length
            

        q = [self.embedding.text_to_sequence(sent) for sent in q]
#        q = to_array(q,maxlen = self.max_sequence_length, use_mask = False)
        a = [self.embedding.text_to_sequence(sent) for sent in a]
#        a = to_array(a,maxlen = self.max_sequence_length, use_mask = False) 
        y = to_categorical(np.asarray(y))
        
        if self.bert_enabled:
            q,q_mask = to_array(q,maxlen = self.max_sequence_length, use_mask = True)
            a,a_mask = to_array(a,maxlen = self.max_sequence_length, use_mask = True)
            x_data = [q,q_mask,a,a_mask]
            if self.match_type == 'pairwise':
                x_data = x_data+[a,a_mask]
                y = [l for l in zip(*[q,a,a])]
            
        else:
            q = to_array(q,maxlen = self.max_sequence_length, use_mask = False)
            a = to_array(a,maxlen = self.max_sequence_length, use_mask = False)
            x_data = [q,a]
            if self.match_type == 'pairwise':
                x_data = x_data+[a]
                y = [l for l in zip(*[q,a,a])]
        if overlap_feature:
            overlap = [self.overlap_index(q_seq,a_seq) for q_seq, a_seq in zip(*x_data)]
            x_data = x_data+overlap

        if iterable:
            x = [l for l in zip(*x_data)]
            data = (x,y)
            return BucketIterator(data,batch_size=self.batch_size, batch_num = int(self.num_samples/self.batch_size),shuffle=True) 
        else: 
            return x_data,y
        
        

    
#    @log_time_delta
    def get_train(self,shuffle = True,model= None,sess= None,overlap_feature= False,iterable=True,max_sequence_length=0):
        
        q,a,neg_a,overlap1,overlap2 = [],[],[],[],[]
        for question,group in self.datas["train"].groupby("question"):
            pos_answers = group[group["flag"] == 1]["answer"]
            neg_answers = group[group["flag"] == 0]["answer"]#.reset_index()
            if len(pos_answers)==0 or len(neg_answers)==0:
    #            print(question)
                continue
            for pos in pos_answers:  
                
                #sampling with model
                if model is not None and sess is not None:                    
                    pos_sent = self.embedding.text_to_sequence(pos)
                    q_sent,q_mask = self.prepare_data([pos_sent])                             
                    neg_sents = [self.embedding.text_to_sequence(sent) for sent in neg_answers]
                    a_sent,a_mask = self.prepare_data(neg_sents)                   
                    scores = model.predict(sess,(np.tile(q_sent,(len(neg_answers),1)),a_sent))
                    neg_index = scores.argmax()   
                    seq_neg_a = neg_sents[neg_index]
                    
                #just random sampling
                else:    
#                    if len(neg_answers.index) > 0:
                    neg_index = np.random.choice(neg_answers.index)
                    neg = neg_answers.loc[neg_index,]
                    seq_neg_a = self.embedding.text_to_sequence(neg)
                
                seq_q = self.embedding.text_to_sequence(question)
                seq_a = self.embedding.text_to_sequence(pos)
                
                q.append(seq_q)
                a.append(seq_a)
                neg_a.append(seq_neg_a)
                if overlap_feature:
                    overlap1.append(self.overlap_index(seq_q,seq_a))
                    overlap2.append(self.overlap_index(seq_q,seq_neg_a))
        if overlap_feature:
            data= (q,a,neg_a,overlap1,overlap2)
        else:
            data = (q,a,neg_a)
#        print("samples size : " +str(len(q)))
        if iterable:
#            data_generator = BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=self.max_sequence_length) 
            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=self.max_sequence_length) 
                        
#            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=max_sequence_length) 
        else: 
            return data
        
    # calculate the overlap_index
    def overlap_index(self,question,answer):

        qset = set(question)
        aset = set(answer)
        a_len = len(answer)
    
        # q_index = np.arange(1,q_len)
        a_index = np.arange(1,a_len + 1)
    
        overlap = qset.intersection(aset)
        for i,a in enumerate(answer):
            if a in overlap:
                a_index[i] = Overlap
        return a_index
            
    
    def get_test(self,overlap_feature = False, iterable = True):
        
        if overlap_feature:
            process = lambda row: [self.embedding.text_to_sequence(row["question"]),
                               self.embedding.text_to_sequence(row["answer"]), 
                               self.embedding.overlap_index(row['question'],row['answer'] )]
        else:
            process = lambda row: [self.embedding.text_to_sequence(row["question"]),
                               self.embedding.text_to_sequence(row["answer"])]
        
        samples = self.datas['test'].apply(process,axis=1)
        if iterable:
            return BucketIterator([i for i in zip(*samples)],batch_size=self.batch_size,shuffle=False)
        else: 
            if self.match_type == 'pointwise':
                
#                [to_array(i,reader.max_sequence_length) for i in test_data]
                return [to_array(i,self.max_sequence_length) for i in zip(*samples)]
            else:
#                return [[i,i] for i in zip(*samples)]
                return [[to_array(i,self.max_sequence_length),to_array(i,self.max_sequence_length)] for i in zip(*samples)]
    

    def batch_gen(self, data_generator):
        if self.match_type == 'pointwise':
#            self.unbalanced_sampling = False
            if self.unbalanced_sampling:
#                print('system goes here!!')
                process = lambda row: [self.embedding.text_to_sequence(row["question"]),
                       self.embedding.text_to_sequence(row["answer"]), 
                       row['flag'] ]
                samples = self.datas["train"].apply(process,axis=1)
                for batch in BucketIterator([i for i in zip(*samples.values)],batch_size=self.batch_size,shuffle=True,max_sequence_length=self.max_sequence_length):
                    if self.onehot:
                        if self.bert_enabled:
                            
                            q,q_mask = to_array(batch[0],self.max_sequence_length,use_mask = True)
                            a,a_mask = to_array(batch[1],self.max_sequence_length,use_mask = True)
                            yield [q,q_mask,a,a_mask], np.array([[0,1] if i else [1,0] for i in batch[2]])
                        else:
                            yield batch[:2],np.array([[0,1] if i else [1,0] for i in batch[2]])
                    else:
                        
                        if self.bert_enabled:
                            q,q_mask = to_array(batch[0],self.max_sequence_length,use_mask = True)
                            a,a_mask = to_array(batch[1],self.max_sequence_length,use_mask = True)
                            yield [q,q_mask,a,a_mask], np.array(batch[2])
                        else:
                            yield batch[:2], np.array(batch[2])
            else:
                while True:
                    for batch in data_generator:
                        q,a,neg = batch
                        if self.onehot:
                            data = [[np.concatenate([q,q],0).astype(int),np.concatenate([a,neg],0).astype(int)],
                                 np.array([[0,1]]*len(q) +[[1,0]]*len(q))]
                        else:
                            data = [[np.concatenate([q,q],0).astype(int),np.concatenate([a,neg],0).astype(int)],
                                     [1]*len(q) +[0]*len(q)]
                        yield data
                                   
        if self.match_type == 'pairwise':
            while True:
                for batch in data_generator:
                    if self.bert_enabled:
                        
                        q,q_mask = to_array(batch[0],self.max_sequence_length,use_mask = True)
                        a,a_mask = to_array(batch[1],self.max_sequence_length,use_mask = True)
                        neg_a, neg_a_mask = to_array(batch[2],self.max_sequence_length,use_mask = True)
                        
                        yield [q,q_mask,a,a_mask,neg_a,neg_a_mask], [q,a,neg_a]
                    else:
                        yield batch,batch
                    

        
            
    def prepare_data(self,seqs):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        max_len = np.max(lengths)
    
        x = np.zeros((n_samples, max_len)).astype('int32')
        x_mask = np.zeros((n_samples, max_len)).astype('float')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
         # print( x, x_mask)
        return x, x_mask
    
    def evaluate(self,predicted,mode= "test",acc=False):
        return evaluation.evaluationBypandas(self.datas[mode],predicted,acc=acc)
        

if __name__ == "__main__":
    
    
#    from dataset import qa
#    from params import Params
#    
#    params = Params()
#    config_file = 'config/qa.ini'    # define dataset in the config
#    params.parse_config(config_file)
#    
#    reader = qa.setup(params)
##    data1 = next(iter(reader.getTest()))
##    data = next(iter(reader.getTrain(overlap_feature=True)))
##    for data in reader.getTest(overlap_feature=True,shuffle=False):
##        print(len(data))
##    data = next(iter(reader.getTrain(overlap_feature=True,shuffle=False)))
##    data = next(iter(reader.getTest(overlap_feature=True)))
#    data = reader.getTrain(iterable=False)
    # -*- coding: utf-8 -*-
    import keras
    from keras.layers import Input, Dense, Activation, Lambda
    import numpy as np
    from keras import regularizers
    from keras.models import Model
    import sys
    from params import Params
    from dataset import qa
    import keras.backend as K
    import units
    from loss import *

    from models.match import keras as models
    from params import Params
    params = Params()

    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    
    reader = qa.setup(params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    from loss import *
    model.compile(loss = rank_hinge_loss({'margin':0.2}),
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
    model.summary()
    
    
    
    
#    generators = [reader.getTrain(iterable=False) for i in range(params.epochs)]
#    [q,a,score] = reader.getPointWiseSamples()
#    model.fit(x = [q,a,a],y = [q,a,q],epochs = 10,batch_size =params.batch_size)
    
#    def gen():
#        while True:
#            for sample in reader.getTrain(iterable = True):
#                yield sample
    model.fit_generator(reader.getPointWiseSamples4Keras(),epochs = 20,steps_per_epoch=1000)

