#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
import math
from scipy import linalg
from numpy.random import RandomState
rng = np.random.RandomState(23455)
from keras import backend as K
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
sys.path.append('../../../')
from modules.interaction.tensorflow.density_represtion import Density_represtion


class complex_cnn(object):
    def __init__(
      self, question,answer,embeddings,max_input_left,max_input_right,is_Embedding_Needed,trainable,dropout_keep_prob,filter_sizes,num_filters,vocab_size,embedding_dim):

        self.question=question
        self.answer=answer
        self.embeddings = embeddings
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.is_Embedding_Needed=is_Embedding_Needed
        self.trainable=trainable
        self.dropout_keep_prob=dropout_keep_prob
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.filter_sizes="2"
        self.rng = 23455

    def density_matrix(self):
        matrix=Density_represtion(self.question,self.answer,self.embeddings,self.max_input_left,self.max_input_right,self.is_Embedding_Needed,self.trainable,self.dropout_keep_prob,self.vocab_size,self.embedding_dim)
        matrix.build_graph()
        self.M_qa_real,self.M_qa_imag=matrix.joint_representation()
    def direct_representation(self):

        self.embedded_q = tf.reshape(self.embedded_chars_q,[-1,self.max_input_left,self.total_embedding_dim])
        self.embedded_a = tf.reshape(self.embedded_chars_a,[-1,self.max_input_right,self.total_embedding_dim])
        reverse_a = tf.transpose(self.embedded_a,[0,2,1])
        self.M_qa = tf.matmul(self.embedded_q,reverse_a)

    def trace_represent(self):
        self.density_diag = tf.matrix_diag_part(self.M_qa)
        self.density_trace = tf.expand_dims(tf.trace(self.M_qa),-1)
        self.match_represent = tf.concat([self.density_diag,self.density_trace],1)
    def convolution(self):
        #initialize my conv kernel
        self.kernels_real = []
        self.kernels_imag=[]
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv_pool"):
                filter_shape = [int(filter_size),int(filter_size),1,self.num_filters]
                input_dim=2
                print(filter_shape[:-1])
                fan_in = np.prod(filter_shape[:-1])
                fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
                s=1./fan_in
                rng=RandomState(23455)
                modulus=rng.rayleigh(scale=s,size=filter_shape)
                phase=rng.uniform(low=-np.pi,high=np.pi,size=filter_shape)
                W_real=modulus*np.cos(phase)
                W_imag=modulus*np.sin(phase)
                W_real = tf.Variable(W_real,dtype = 'float32')
                W_imag = tf.Variable(W_imag,dtype = 'float32')
                self.kernels_real.append(W_real)
                self.kernels_imag.append(W_imag)
                # self.para.append(W_real)
                # self.para.append(W_imag)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.qa_real = self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)-self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)
        print("qa_real")
        self.qa_imag = self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)+self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)
    def max_pooling(self,conv):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, self.max_input_left, self.max_input_right, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def avg_pooling(self,conv):
        pooled = tf.nn.avg_pool(
                    conv,
                    ksize = [1, self.max_input_left, self.max_input_right, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def pooling_graph(self):
        with tf.name_scope('pooling'):

      
            raw_pooling_real = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_real,1))
            col_pooling_real = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_real,2))
            self.represent_real = tf.concat([raw_pooling_real,col_pooling_real],1)
            
            raw_pooling_imag = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_imag,1))
            col_pooling_imag = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_imag,2))
            self.represent_imag = tf.concat([raw_pooling_imag,col_pooling_imag],1)
            print("self.represent_real,self.represent_imag")
            return self.represent_real,self.represent_imag
    
    def wide_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv-1"
            )
            cnn_outputs.append(conv)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            cnn_outputs.append(conv)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        # return cnn_outputs
        return cnn_reshaped
    def build_graph(self):
        self.density_matrix()
        self.convolution()
        self.pooling_graph()

if __name__ == '__main__':
    cnn = QA_quantum(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                embeddings_complex=None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3*33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y
        }

        scores = sess.run([cnn.scores],feed_dict)
        print (scores)

