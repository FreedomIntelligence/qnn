#-*- coding:utf-8 -*-

import os
import numpy as np
import tensorflow as tf

from collections import Counter
import pandas as pd

from tqdm import tqdm
import random
import pickle
from utils import log_time_delta

from nltk.corpus import stopwords


OVERLAP = 237
class Alphabet(dict):
    def __init__(self, start_feature_id = 1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))
                
def cut(sentence):
    
    tokens = sentence.lower().split()
    # tokens = [w for w in tokens if w not in stopwords.words('english')]
    return tokens

@log_time_delta
def load(dataset, filter = False):
    data_dir = "data/" + dataset
    datas = []
    for data_name in ['train.txt','test.txt','dev.txt']:
        data_file = os.path.join(data_dir,data_name)
        data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"]).fillna('0')
#        data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('0')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    # sub_file = os.path.join(data_dir,'submit.txt')
    # submit = pd.read_csv(sub_file,header = None,sep = "\t",names = ['question','answer'],quoting = 3)
    # datas.append(submit)
    return tuple(datas)
@log_time_delta
def removeUnanswerdQuestion(df):
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct=counter[counter>0].index
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]==0))
    questions_have_uncorrect=counter[counter>0].index
    counter=df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi=counter[counter>1].index

    return df[df["question"].isin(questions_have_correct) &  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()
@log_time_delta
def get_alphabet(corpuses=None,dataset=""):
    pkl_name="temp/"+dataset+".alphabet.pkl"
    if  os.path.exists(pkl_name):
        return pickle.load(open(pkl_name,"rb"))
    alphabet = Alphabet(start_feature_id = 0)
    alphabet.add('[UNK]')  
    alphabet.add('END') 
    count = 0
    for corpus in corpuses:
        for texts in [corpus["question"].unique(),corpus["answer"]]:

            for sentence in texts:                   
                tokens = cut(sentence)
                for token in set(tokens):
                    alphabet.add(token)
        print("alphabet size %d" % len(alphabet.keys()) )
    if not os.path.exists("temp"):
        os.mkdir("temp")
    pickle.dump( alphabet,open(pkl_name,"wb"))
    return alphabet
@log_time_delta
def getSubVectorsFromDict(vectors,vocab,dim = 300):
    embedding = np.zeros((len(vocab),dim))
    count = 1
    for word in vocab:
        if word in vectors:
            count += 1
            embedding[vocab[word]]= vectors[word]
        else:
            embedding[vocab[word]]= np.random.uniform(-0.5,+0.5,dim)#vectors['[UNKNOW]'] #.tolist()
    print( 'word in embedding',count)
    return embedding
def encode_to_split(sentence,alphabet):
    indices = []    
    tokens = cut(sentence)
    seq = [alphabet[w] if w in alphabet else alphabet['[UNK]'] for w in tokens]
    return seq
@log_time_delta
def load_text_vec(alphabet,filename="",embedding_size = 100):
    vectors = {}
    with open(filename,encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print( 'epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print( ( vocab_size, embedding_size))
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print( 'embedding_size',embedding_size)
    print( 'done')
    print( 'words found in wor2vec embedding ',len(vectors.keys()))
    return vectors
@log_time_delta
def get_embedding(alphabet,dim = 300,language ="en",dataset=""):
    pkl_name="temp/"+dataset+".subembedding.pkl"
    if  os.path.exists(pkl_name):
        return pickle.load(open(pkl_name,"rb"))
    if language=="en":
        fname = 'embedding/glove.6B/glove.6B.300d.txt'
    else:
        fname= "embedding/embedding.200.header_txt"
    embeddings = load_text_vec(alphabet,fname,embedding_size = dim)
    sub_embeddings = getSubVectorsFromDict(embeddings,alphabet,dim)
    pickle.dump( sub_embeddings,open(pkl_name,"wb"))
    return sub_embeddings

@log_time_delta
def get_mini_batch_test(df,alphabet,batch_size):
    q = []
    a = []
    pos_overlap = []
    for index,row in df.iterrows():
        question = encode_to_split(row["question"],alphabet)
        answer = encode_to_split(row["answer"],alphabet)
        overlap_pos = overlap_index(row['question'],row['answer'])
        q.append(question)
        a.append(answer)
        pos_overlap.append(overlap_pos)

    m = 0
    n = len(q)
    idx_list = np.arange(m,n,batch_size)
    mini_batches = []
    for idx in idx_list:
        mini_batches.append(np.arange(idx,min(idx + batch_size,n)))
    for mini_batch in mini_batches:
        mb_q = [ q[t] for t in mini_batch]
        mb_a = [ a[t] for t in mini_batch]
        mb_pos_overlap = [pos_overlap[t] for t in mini_batch]
        mb_q,mb_q_mask = prepare_data(mb_q)
        mb_a,mb_pos_overlaps = prepare_data(mb_a,mb_pos_overlap)


        yield(mb_q,mb_a)

# calculate the overlap_index
def overlap_index(question,answer,stopwords = []):
    ans_token = cut(answer)
    qset = set(cut(question))
    aset = set(ans_token)
    a_len = len(ans_token)

    # q_index = np.arange(1,q_len)
    a_index = np.arange(1,a_len + 1)

    overlap = qset.intersection(aset)
    # for i,q in enumerate(cut(question)[:q_len]):
    #     value = 1
    #     if q in overlap:
    #         value = 2
    #     q_index[i] = value
    for i,a in enumerate(ans_token):
        if a in overlap:
            a_index[i] = OVERLAP
    return a_index



def getBatch48008(df,alphabet,batch_size,sort_by_len = True,shuffle = False):
    q,a,neg_a=[],[],[]
    answers=df["answer"][:250]
    ground_truth=df.groupby("question").apply(lambda group: group[group.flag==1].index[0]%250 ).to_dict() 
    
    for question in tqdm(df['question'].unique()):
                   
        index= ground_truth[question]  
        
        canindates = [i for i in range(250)]
        canindates.remove(index)
        a_neg_index = random.choice(canindates)

        seq_q = encode_to_split(question,alphabet)
        seq_a = encode_to_split(answers[index],alphabet)
        seq_neg_a = encode_to_split(answers[a_neg_index],alphabet)
        
        q.append(seq_q)       
        a.append( seq_a)
        neg_a.append(seq_neg_a )
        
    return iteration_batch(q,a,neg_a,batch_size,sort_by_len,shuffle)    
def iteration_batch(q,a,neg_a,batch_size,sort_by_len = True,shuffle = False):


    if sort_by_len:
        sorted_index = sorted(range(len(q)), key=lambda x: len(q[x]), reverse=True)
        q = [ q[i] for i in sorted_index]
        a = [a[i] for i in sorted_index]
        neg_a = [ neg_a[i] for i in sorted_index]

        pos_overlap = [pos_overlap[i] for i in sorted_index]
        neg_overlap = [neg_overlap[i] for i in sorted_index]

    #get batch
    m = 0
    n = len(q)

    idx_list = np.arange(m,n,batch_size)
    if shuffle:
        np.random.shuffle(idx_list)

    mini_batches = []
    for idx in idx_list:
        mini_batches.append(np.arange(idx,min(idx + batch_size,n)))

    for mini_batch in tqdm(mini_batches):
        mb_q = [ q[t] for t in mini_batch]
        mb_a = [ a[t] for t in mini_batch]
        mb_neg_a = [ neg_a[t] for t in mini_batch]
        mb_pos_overlap = [pos_overlap[t] for t in mini_batch]
        mb_neg_overlap = [neg_overlap[t] for t in mini_batch]
        mb_q,mb_q_mask = prepare_data(mb_q)
        mb_a,mb_pos_overlaps = prepare_data(mb_a,mb_pos_overlap)
        mb_neg_a,mb_neg_overlaps = prepare_data(mb_neg_a,mb_neg_overlap)
        # mb_a,mb_a_mask = prepare_data(mb_a,mb_pos_overlap)

        # mb_neg_a , mb_a_neg_mask = prepare_data(mb_neg_a)


        yield(mb_q,mb_a,mb_neg_a,mb_q_mask,mb_a_mask,mb_a_neg_mask)


def get_mini_batch(df,alphabet,batch_size,sort_by_len = True,shuffle = False,model=None,sess=None):
    q = []
    a = []
    neg_a = []
    for question in df['question'].unique():
#        group = df[df["question"]==question]
#        pos_answers = group[df["flag"] == 1]["answer"]
#        neg_answers = group[df["flag"] == 0]["answer"].reset_index()
        group = df[df["question"]==question]
        pos_answers = group[group["flag"] == 1]["answer"]
        neg_answers = group[group["flag"] == 0]["answer"]#.reset_index()

        for pos in pos_answers:
            
            if model is not None and sess is not None:
                
                pos_sent= encode_to_split(pos,alphabet)
                q_sent,q_mask= prepare_data([pos_sent])
                
                neg_sents = [encode_to_split(sent,alphabet) for sent in neg_answers] 

                a_sent,a_mask= prepare_data(neg_sents)                    
  
                scores = model.predict(sess,(np.tile(q_sent,(len(neg_answers),1)),a_sent,np.tile(q_mask,(len(neg_answers),1)),a_mask))
                neg_index = scores.argmax()
          

               
            else:

                if len(neg_answers.index) > 0:
                    neg_index = np.random.choice(neg_answers.index)
            neg = neg_answers.reset_index().loc[neg_index,]["answer"]
            seq_q = encode_to_split(question,alphabet)
            seq_a = encode_to_split(pos,alphabet)
            seq_neg_a = encode_to_split(neg,alphabet)
            q.append(seq_q)
            a.append(seq_a)
            neg_a.append(seq_neg_a)
    return iteration_batch(q,a,neg_a,batch_size,sort_by_len,shuffle)
    

def prepare_data(seqs,overlap = None):

    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples,max_len)).astype('int32')
    if overlap is not None:
        overlap_position = np.zeros((n_samples,max_len)).astype('float')

        for idx ,seq in enumerate(seqs):
            x[idx,:lengths[idx]] = seq
            overlap_position[idx,:lengths[idx]] = overlap[idx]
        return x,overlap_position
    else:
        x_mask = np.zeros((n_samples, max_len)).astype('float')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
        # print( x, x_mask)
        return x, x_mask

# def prepare_data(seqs):
#     lengths = [len(seq) for seq in seqs]
#     n_samples = len(seqs)
#     max_len = np.max(lengths)

#     x = np.zeros((n_samples, max_len)).astype('int32')
#     x_mask = np.zeros((n_samples, max_len)).astype('float')
#     for idx, seq in enumerate(seqs):
#         x[idx, :lengths[idx]] = seq
#         x_mask[idx, :lengths[idx]] = 1.0
#     # print( x, x_mask)
#     return x, x_mask
    









