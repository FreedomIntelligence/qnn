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
from models.match.tensorflow.QA_CNN import complex_cnn


class QA_quantum(object):
    def __init__(self, opt):
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)
            print("%s:%s" %(key,value))
        # self.dropout_keep_prob = dropout_keep_prob
        # self.num_filters = num_filters
        # self.embeddings = self.lookuptable
        # self.embeddings_complex=embeddings_complex
        # self.embedding_size = embedding_size
        # self.overlap_needed = overlap_needed
        # self.vocab_size = vocab_size
        # self.trainable = trainable
        # self.filter_sizes = filter_sizes
        # self.pooling = pooling
        # self.position_needed = position_needed
        # if self.overlap_needed:
        #     self.total_embedding_dim = embedding_size + extend_feature_dim
        # else:
        #     self.total_embedding_dim = embedding_size
        # if self.position_needed:
        #     self.total_embedding_dim = self.total_embedding_dim + extend_feature_dim
        # self.batch_size = batch_size
        # self.l2_reg_lambda = l2_reg_lambda
        # self.para = []
        # self.max_input_left = max_input_left
        # self.max_input_right = max_input_right
        # self.hidden_num = self.num_filters
        # self.extend_feature_dim = extend_feature_dim
        # self.is_Embedding_Needed = is_Embedding_Needed
        self.embeddings = self.lookup_table
        self.vocab_size=self.embeddings.shape[0]
        self.embedding_dim=50
        self.is_Embedding_Needed=True
        self.rng = 23455
        self.filter_sizes="2"
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[None,None],name = 'input_question')
        self.batch_size = tf.shape(self.question)[0]
        self.answer = tf.placeholder(tf.int32,[None,None],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")

    def set_weight(self,num_unit,dim):
        input_dim = (self.embedding_dim - int(self.filter_sizes[0]) + 1) * int(self.num_filters) * dim
        unit=num_unit
        kernel_shape = [input_dim,unit]
        fan_in_f=np.prod(kernel_shape)
        s = np.sqrt(1. / fan_in_f)
        rng=RandomState(23455)
        modulus_f=rng.rayleigh(scale=s,size=kernel_shape)
        phase_f=rng.uniform(low=-np.pi,high=np.pi,size=kernel_shape)
        real_init=modulus_f*np.cos(phase_f)
        imag_init=modulus_f*np.sin(phase_f)
        real_kernel=tf.Variable(real_init,name='real_kernel')
        real_kernel=tf.to_float(real_kernel)
        imag_kernel=tf.Variable(imag_init,name='imag_kernel')
        imag_kernel=tf.to_float(imag_kernel)
        return real_kernel,imag_kernel
    def feed_neural_work(self):
        with tf.name_scope('regression'):
            cnn_feature=complex_cnn(self.question,self.answer,self.embeddings,self.max_sequence_length,self.max_sequence_length,self.is_Embedding_Needed,self.embedding_trainable,self.dropout_keep,self.filter_sizes,self.num_filters,self.vocab_size,self.embedding_dim)
            cnn_feature.build_graph()
            self.represent_real,self.represent_imag=cnn_feature.pooling_graph()
            self.represent=tf.concat([self.represent_imag,self.represent_real],1)
            dim_list=self.represent_real.shape.as_list()
            self.real_kernel_1,self.imag_kernel_1=self.set_weight(int(dim_list[1]/2),2)
            self.full_join_real_1=tf.matmul(self.represent_real,self.real_kernel_1)-tf.matmul(self.represent_imag,self.imag_kernel_1)
            self.full_join_imag_1=tf.matmul(self.represent_real,self.imag_kernel_1)+tf.matmul(self.represent_imag,self.real_kernel_1)
            self.real_kernel_2,self.imag_kernel_2=self.set_weight(1,1)
            self.full_join_real_2=tf.matmul(self.full_join_real_1,self.real_kernel_2)-tf.matmul(self.full_join_imag_1,self.imag_kernel_2)
            self.full_join_imag_2=tf.matmul(self.full_join_real_1,self.imag_kernel_2)+tf.matmul(self.full_join_imag_1,self.real_kernel_2)
            b = tf.get_variable('b_hidden', shape=[2],initializer = tf.random_normal_initializer())
            print("self.logits")
            self.logits=tf.concat([self.full_join_real_2,self.full_join_imag_2],1)+b
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
    def create_loss(self):
        l2_loss = tf.constant(0.0)
        # for p in self.para:
        #     l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            #pi_regularization = tf.reduce_sum(self.weighted_q) - 1 + tf.reduce_sum(self.weighted_a) - 1
            #self.loss = tf.reduce_mean(losses)+self.l2_reg_lambda*l2_loss+0.0001*tf.nn.l2_loss(pi_regularization)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    def train(self,sess,data):
        print(data)
        feed_dict = {
                self.question:data[0],
                self.answer:data[1],
                self.input_y:data[2]
#                self.pos_position:data[3],
#                self.neg_position:data[4],
                # self.q_mask:data[3],
                # self.a_mask:data[4],
                # self.a_neg_mask:data[5],
            }
        print(11111)

        _, step, loss, accuracy = sess.run(
                    [self.train_op,self.global_step,self.loss, self.accuracy],
                    feed_dict)
        print("step {}, loss {:g}, acc {:g} ".format(step, loss, accuracy))
        print(1)
        return _, step, loss, accuracy
    def create_op(self):
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step = self.global_step)

    def build_graph(self):
        self.create_placeholder()
        self.feed_neural_work()
        self.create_loss()
        self.create_op()