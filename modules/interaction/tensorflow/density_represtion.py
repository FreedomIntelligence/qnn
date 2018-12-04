import tensorflow as tf
import numpy as np
#from multiply import ComplexMultiply
import math
from scipy import linalg
# point_wise obbject
from numpy.random import RandomState
rng = np.random.RandomState(23455)
from keras import initializers
import sys
sys.path.append('../../../')
from modules.embedding.tensorflow.ComplexWordEmbedding import ComplexEmbedding
#from complexnn.dense import ComplexDense
#from complexnn.utils import GetReal
class Density_represtion(object):
	def __init__(
      self, question,answer,embeddings,max_input_left,max_input_right,is_Embedding_Needed,trainable,dropout_keep_prob,vocab_size,embedding_dim):
		self.question=question
		self.answer=answer
		self.embeddings = embeddings
		self.max_input_left = max_input_left
		self.max_input_right = max_input_right
		self.is_Embedding_Needed=is_Embedding_Needed
		self.trainable=trainable
		self.dropout_keep_prob=dropout_keep_prob
		self.vocab_size=vocab_size
		self.embedding_dim=embedding_dim
		self.rng = 23455
	def density_weighted(self):
		self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]) , name = 'weighted_q')
		self.weighted_q=tf.nn.softmax(self.weighted_q,1)
		#self.para.append(self.weighted_q)
		self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]) , name = 'weighted_a')
		self.weighted_a=tf.nn.softmax(self.weighted_a,1)
		#self.para.append(self.weighted_a)
	def add_embeddings(self):
		with tf.name_scope("embedding"):
			embedding_complex=ComplexEmbedding(self.vocab_size,self.embedding_dim)
			self.embeddings_complex=embedding_complex.getSubVectors_complex_random()
			W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
			W_complex=tf.Variable(np.array(self.embeddings_complex),name="W" ,dtype="float32",trainable = self.trainable)
			self.embedding_W = W
			self.embedding_W_complex=W_complex
		self.embedded_chars_q,self.embedding_chars_q_complex = self.concat_embedding(self.question)
		self.embedded_chars_a,self.embedding_chars_a_complex= self.concat_embedding(self.answer)
	def joint_representation(self):
		self.density_q_real,self.density_q_imag = self.density_matrix(self.embedded_chars_q,self.embedding_chars_q_complex,self.weighted_q)
		self.density_a_real,self.density_a_imag = self.density_matrix(self.embedded_chars_a,self.embedding_chars_a_complex,self.weighted_a)
		self.M_qa_real=tf.matmul(self.density_q_real,self.density_a_real)+tf.matmul(self.density_q_imag,self.density_a_imag)
		self.M_qa_imag=tf.matmul(self.density_q_imag,self.density_a_real)-tf.matmul(self.density_q_real,self.density_a_imag)
		return self.M_qa_real,self.M_qa_imag
	def density_matrix(self,sentence_matrix,sentence_matrix_complex,sentence_weighted):
		self.input_real=tf.expand_dims(sentence_matrix,-1)
		self.input_imag=tf.expand_dims(sentence_matrix_complex,-1)
		input_real_transpose = tf.transpose(self.input_real, perm = [0,1,3,2])
		input_imag_transpose = tf.transpose(self.input_imag, perm = [0,1,3,2])
		q_a_real_real = tf.matmul(self.input_real,input_real_transpose)
		q_a_real_imag = tf.matmul(self.input_imag,input_imag_transpose)
		q_a_real = q_a_real_real-q_a_real_imag
		q_a_imag_real=tf.matmul(self.input_imag,input_real_transpose)
		q_a_imag_imag=tf.matmul(self.input_real,input_imag_transpose)
		q_a_imag = q_a_imag_real+q_a_imag_imag
		return tf.reduce_sum(tf.multiply(q_a_real,sentence_weighted),1),tf.reduce_sum(tf.multiply(q_a_imag,sentence_weighted),1)
	def concat_embedding(self,words_indice):
		embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
		embedded_chars_q=tf.nn.dropout(embedded_chars_q, self.dropout_keep_prob, name="hidden_output_drop")
		embedding_chars_q_complex=tf.nn.embedding_lookup(self.embedding_W_complex,words_indice)
		embedding_chars_q_complex=tf.nn.dropout(embedding_chars_q_complex, self.dropout_keep_prob, name="hidden_output_drop")
		# [embedded_chars_q, embedding_chars_q_complex] = ComplexMultiply()([embedding_chars_q_complex,embedded_chars_q])
		return embedded_chars_q,embedding_chars_q_complex
	def build_graph(self):
		self.add_embeddings()
		self.density_weighted()
		self.joint_representation()
