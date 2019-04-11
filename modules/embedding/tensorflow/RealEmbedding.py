import numpy as np
import random,os,math
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
class RealEmbedding: 
    def getSubVectors(vectors, vocab, dim=50):
        print ('embedding_size:', vectors.syn0.shape[1])
        embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
        temp_vec = 0
        for word in vocab:
            if word in vectors.vocab:
                embedding[vocab[word]] = vectors.word_vec(word)
            else:
                # .tolist()
                embedding[vocab[word]
                          ] = np.random.uniform(-0.25,+0.25,vectors.syn0.shape[1])
            temp_vec += embedding[vocab[word]]
        temp_vec /= len(vocab)
        for index, _ in enumerate(embedding):
            embedding[index] -= temp_vec
        return embedding