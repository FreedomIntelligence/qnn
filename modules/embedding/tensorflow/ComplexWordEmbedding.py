import numpy as np
import random,os,math
import pandas as pd
import warnings
class ComplexEmbedding(object):
	def __init__(self,vocab_size,dim):
		self.vocab_size=vocab_size
		self.dim=dim

	def getSubVectors_complex_random(self):
		embedding = np.zeros((self.vocab_size, self.dim))
		for i in range(self.vocab_size):
		    embedding[i] = np.random.uniform(0, +2*math.pi, self.dim)
		return embedding