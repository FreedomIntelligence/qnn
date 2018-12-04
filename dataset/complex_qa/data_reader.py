# -*- coding: utf-8 -*-

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

Overlap = 237
import random
from units import to_array 
from tools import evaluation
from preprocess.dictionary import Dictionary
from preprocess.embedding import Embedding
from preprocess.bucketiterator import BucketIterator

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
        
        if 'bert' in self.network_type:
            loaded_dic = Dictionary(dict_path =self.dict_path)
            self.embedding = Embedding(loaded_dic,self.max_sequence_length)
        else:
            self.embedding = Embedding(self.get_dictionary(self.datas.values()),self.max_sequence_length)
            
        self.embedding = Embedding(self.get_dictionary(self.datas.values()),self.max_sequence_length)
        self.alphabet=self.get_dictionary(self.datas.values())

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
    
    
#    @log_time_delta
    def transform(self,flag):
        if flag == 1:
            return [0, 1]
        else:
            return [1, 0]
    def cut(self,sentence, isEnglish=True):
        if isEnglish:
            tokens =sentence.lower().split()
        else:
            # words = jieba.cut(str(sentence))
            tokens = [word for word in sentence.split() if word not in stopwords]
        return tokens
    def encode_to_split(self,sentence, max_sentence):
        indices = []
        tokens = self.cut(sentence)
        for word in tokens:
            indices.append(self.alphabet[word])
        while(len(indices) < max_sentence):
            indices += indices[:(max_sentence - len(indices))]
        # results=indices+[alphabet["END"]]*(max_sentence-len(indices))
        return indices[:max_sentence]
    def get_train(self,overlap_feature=False):
        input_num = 3
        pairs = []
        print(self.datas["train"]["question"][0]==self.datas["train"]["question"][1])
        for i in range(len(self.datas["train"])):
            print(i)
            if self.datas["train"]["question"][i]==self.datas["train"]["question"][i+1] and self.datas["train"]["flag"][i]==1:
                label_pos=self.transform(1)
                seq_pos_a=self.encode_to_split(self.datas["train"]["answer"][i],max_sentence=self.max_sequence_length)
                question_seq=self.encode_to_split(self.datas["train"]["question"][i],max_sentence=self.max_sequence_length)
                pairs.append((question_seq,seq_pos_a,label_pos))
            if self.datas["train"]["question"][i]==self.datas["train"]["question"][i+1] and self.datas["train"]["flag"][i]==0:
                label_pos=self.transform(0)
                seq_pos_a=self.encode_to_split(self.datas["train"]["answer"][i],max_sentence=self.max_sequence_length)
                question_seq=self.encode_to_split(self.datas["train"]["question"][i],max_sentence=self.max_sequence_length)
                pairs.append((question_seq,seq_pos_a,label_pos))
            else:
                continue
        print(pairs)
        exit()

        for question,group in self.datas["train"].groupby("question"):
            print(question)
            exit()
            pos_answer=group[group["flag"]==1]["answer"]
            neg_answer=group[group["flag"]==0]["answer"]
            if len(pos_answer)==0 or len(neg_answer)==0:
                continue
            for pos in pos_answer:
                neg_index=np.random.choice(neg_answer.index)
                neg=neg_answer.loc[neg_index,]
                label_neg=self.transform(0)
                seq_neg_a=self.encode_to_split(neg,max_sentence=self.max_sequence_length)
                seq_pos_a=self.encode_to_split(pos,max_sentence=self.max_sequence_length)
                label_pos=self.transform(1)
                question_seq=self.encode_to_split(question,max_sentence=self.max_sequence_length)
                pairs.append((question_seq,seq_neg_a,label_neg))
                pairs.append((question_seq,seq_pos_a,label_pos))
        n_batches = int(len(pairs) * 1.0 / self.batch_size)
        pairs = sklearn.utils.shuffle(pairs, random_state=121)

        for i in range(0, n_batches):
            batch = pairs[i * self.batch_size:(i + 1) * self.batch_size]
            yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
        batch = pairs[n_batches * self.batch_size:] + [pairs[n_batches *
                                                        self.batch_size]] * (self.batch_size - len(pairs) + n_batches * self.batch_size)
        yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
#     def get_train(self,shuffle = True,model= None,sess= None,overlap_feature= False,iterable=True,max_sequence_length=0):
        
#         q,a,neg_a,overlap1,overlap2 = [],[],[],[],[]
#         for question,group in self.datas["train"].groupby("question"):
#             pos_answers = group[group["flag"] == 1]["answer"]
#             neg_answers = group[group["flag"] == 0]["answer"]#.reset_index()
#             if len(pos_answers)==0 or len(neg_answers)==0:
#     #            print(question)
#                 continue
#             for pos in pos_answers:  
                
#                 #sampling with model
#                 if model is not None and sess is not None:                    
#                     pos_sent = self.embedding.text_to_sequence(pos)
#                     q_sent,q_mask = self.prepare_data([pos_sent])                             
#                     neg_sents = [self.embedding.text_to_sequence(sent) for sent in neg_answers]
#                     a_sent,a_mask = self.prepare_data(neg_sents)                   
#                     scores = model.predict(sess,(np.tile(q_sent,(len(neg_answers),1)),a_sent))
#                     neg_index = scores.argmax()   
#                     seq_neg_a = neg_sents[neg_index]
                    
#                 #just random sampling
#                 else:    
# #                    if len(neg_answers.index) > 0:
#                     neg_index = np.random.choice(neg_answers.index)
#                     neg = neg_answers.loc[neg_index,]
#                     seq_neg_a = self.embedding.text_to_sequence(neg)
                
#                 seq_q = self.embedding.text_to_sequence(question)
#                 seq_a = self.embedding.text_to_sequence(pos)
                
#                 # q.append(seq_q)
#                 # a.append(seq_a)
#                 # neg_a.append(seq_neg_a)
#                 label_neg=transform(0)
#                 label_pos=transform(1)
#                 if overlap_feature:
#                     overlap1.append(self.overlap_index(seq_q,seq_a))
#                     overlap2.append(self.overlap_index(seq_q,seq_neg_a))
#         if overlap_feature:
#             data= (q,a,neg_a,overlap1,overlap2)
#         else:
#             data.append((seq_q,seq_a,label_pos))

# #        print("samples size : " +str(len(q)))
#         if iterable:
# #            data_generator = BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=self.max_sequence_length) 
#             return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=self.max_sequence_length) 
                        
# #            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=max_sequence_length) 
#         else: 
#             return data
        
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
        
        samples = self.datas['test'].apply( process,axis=1)
        if iterable:
            return BucketIterator([i for i in zip(*samples)],batch_size=self.batch_size,shuffle=False)
        else: 
            if self.match_type == 'pointwise':
                return [i for i in zip(*samples)]
            else:
                return [[i,i] for i in zip(*samples)]
    

    def batch_gen(self, data_generator):
        if self.match_type == 'pointwise':
#            self.unbalanced_sampling = False
            if self.unbalanced_sampling == True:
                process = lambda row: [self.embedding.text_to_sequence(row["question"]),
                       self.embedding.text_to_sequence(row["answer"]), 
                       row['flag'] ]
                samples = self.datas["train"].apply(process,axis=1)
                for batch in BucketIterator( [i for i in zip(*samples)],batch_size=self.batch_size,shuffle=True,max_sequence_length=self.max_sequence_length):
                        if self.onehot:
                            yield batch[:2],np.array([[0,1] if i else [1,0] for i in batch[2]])
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
                    yield batch, batch
                    

        
            
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

