# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from gensim.models.keyedvectors import KeyedVectors
class Embedding(object):
    def __init__(self, dictionary,max_sequence_length):
        self.dictionary = dictionary
        self.embedding_size = 0
        self.max_sequence_length = max_sequence_length
#    @log_time_delta
    def get_embedding(self, dataset_name, fname = None,language ="en", fresh = True):
        pkl_name="temp/"+dataset_name+".subembedding.pkl"
        if  os.path.exists(pkl_name) and not fresh:
            return pickle.load(open(pkl_name,"rb"))
#        if language=="en":
#            fname = 'embedding/glove.6B/glove.6B.300d.txt'
#        else:
#            fname= "embedding/embedding.200.header_txt"
        if fname.endswith("bin"):
            
            embeddings_raw = KeyedVectors.load_word2vec_format(fname, binary=True)
            embeddings={x:y for x,y in zip(embeddings_raw.vocab,embeddings_raw.vectors)}
            embedding_size=embeddings_raw.vectors.shape[1]
        else:
            embeddings,embedding_size = self.load_text_vec(fname)
            
        sub_embeddings = self.get_subVectors(embeddings,embedding_size)
        self.embedding_size=embedding_size
        pickle.dump(sub_embeddings,open(pkl_name,"wb"))
        self.lookup_table = sub_embeddings
        return sub_embeddings
    
    def get_subVectors(self,vectors,dim = 300):
        vocab= self.dictionary
        embedding = np.zeros((len(vocab),dim))
        count = 1
        import codecs
        with codecs.open("oov.txt","w",encoding="utf-8") as f:
            for word in vocab:
                if word in vectors:
                    count += 1
                    embedding[vocab[word]]= vectors[word]
                else:
                    f.write(word+"\n")
                    embedding[vocab[word]]= np.random.uniform(-0.5,+0.5,dim)#vectors['[UNKNOW]'] #.tolist()
        print( 'word in embedding',count)
        print( 'word not in embedding',len(vocab)-count)
        return embedding
    
#    @log_time_delta
    def load_text_vec(self,filename=""):
        vectors = {}
        with open(filename,encoding='utf-8') as f:
            i = 0
            for line in f:
                i += 1
                if i % 100000 == 0:
                    print( 'epoch %d' % i)
                items = line.strip().split(' ')
                if len(items) == 2:
                    vocab_size, embedding_size= items[0],items[1]
                    print( ( vocab_size, embedding_size))
                else:
                    word = items[0]
                    if word in self.dictionary:
                        vectors[word] = items[1:]
        embedding_size = len(items[1:])
        print( 'embedding_size = {}'.format(embedding_size))
        print( 'done.')
        print( '{} words found in word2vec embedding.'.format(len(vectors.keys())))
        return vectors,embedding_size
    
    def text_to_sequence(self,sentence):    
        
        tokens = sentence.lower().split()[:self.max_sequence_length]   # tokens = [w for w in tokens if w not in stopwords.words('english')]
        seq = [self.dictionary[w] if w in self.dictionary else self.dictionary['[UNK]'] for w in tokens]

        return seq

   