#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from tools.timer import log_time_delta

from layers.tensorflow import blocks
# model_type :apn or qacnn
class QARNN(object):

    def __init__(self,opt):
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)
#            print("%s:%s" %(key,value))

        self.position_embedding_dim = 10
        self.attention = 'position_attention'
        self.total_num_filter = len(self.filter_sizes) * self.num_filters
        self.para = []
        
        self.dropout_keep_prob_holder =  tf.placeholder(tf.float32,name = 'dropout_keep_prob')
        print("build over")
    def create_placeholder(self):
        print(('Create placeholders'))
        # he length of the sentence is varied according to the batch,so the None,None
        self.question = tf.placeholder(tf.int32,[None,None],name = 'input_question')
        self.max_input_left = tf.shape(self.question)[1]
   
        self.batch_size_tf = tf.shape(self.question)[0]
        self.answer = tf.placeholder(tf.int32,[None,None],name = 'input_answer')
        self.max_input_right = tf.shape(self.answer)[1]
        self.answer_negative = tf.placeholder(tf.int32,[None,None],name = 'input_right')
        self.pos_position = tf.placeholder(tf.int32,[None,None],name = 'pos_position')
        self.neg_position = tf.placeholder(tf.int32,[None,None],name = 'neg_position')
        self.q_len,self.q_mask = blocks.length(self.question)
        self.a_len,self.a_mask = blocks.length(self.answer)
        self.a_neg_len,self.a_neg_mask = blocks.length(self.answer_negative)
        # self.q_mask = tf.placeholder(tf.int32,[None,None],name = 'q_mask')
        # self.a_mask = tf.placeholder(tf.int32,[None,None],name = 'a_mask')
        # self.a_neg_mask = tf.placeholder(tf.int32,[None,None],name = 'a_neg_mask')
        # #real length
        # self.q_len = tf.reduce_sum(tf.cast(self.q_mask,'int32'),1)
        # self.a_len = tf.reduce_sum(tf.cast(self.a_mask,'int32'),1)
        # self.a_neg_len = tf.reduce_sum(tf.cast(self.a_neg_mask,'int32'),1)
    def create_position(self):
        self.a_max_len = tf.shape(self.answer)[1]
        self.a_neg_max_len = tf.shape(self.answer_negative)[1]
        self.a_position = tf.tile(tf.reshape(tf.range(self.a_max_len),[1,self.a_max_len]),[self.batch_size_tf,1],name="a_position")
        self.a_neg_position = tf.tile(tf.reshape(tf.range(self.a_neg_max_len),[1,self.a_neg_max_len]),[self.batch_size_tf,1],name="a_neg_position")
    
    def add_embeddings(self):
        print( 'add embeddings')
        if self.lookup_table is not None:
            print( "load embedding")
            W = tf.Variable(np.array(self.lookup_table),name = "embedding" ,dtype="float32",trainable = self.embedding_trainable)
            
        else:
            print( "random embedding")
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="embedding",trainable = self.embedding_trainable)
        self.embedding_W = W
        self.position_embedding = tf.Variable(tf.random_uniform([300,self.position_embedding_dim],-1.0,1.0),name = 'position_embedding')
       
        # self.overlap_W = tf.Variable(a,name="W",trainable = True)
        self.para.append(self.embedding_W)

        self.q_embedding =  tf.nn.embedding_lookup(self.embedding_W,self.question,name="q_embedding")
        self.a_embedding = tf.nn.embedding_lookup(self.embedding_W,self.answer,name="a_embedding")
        self.a_neg_embedding = tf.nn.embedding_lookup(self.embedding_W,self.answer_negative,name="a_neg_embedding")
        

        self.a_p = tf.nn.embedding_lookup(self.position_embedding ,self.a_position, name="a_position_embedding")
        self.a_neg_p = tf.nn.embedding_lookup(self.position_embedding,self.a_neg_position,name="a_neg_position_embedding")

    def rnn_model_sentence(self):
        fw_cell,bw_cell = self.lstm_cell('rnn')
        self.para_initial()
        if self.attention == 'iarnn_word':
            self.rnn_att_inner(fw_cell,bw_cell)
        elif self.attention == 'position_attention':
            self.rnn_position_attention(fw_cell,bw_cell)
        else:
            self.rnn_att_tra(fw_cell,bw_cell)
            
        # print( self.q_pos_rnn)
        # self.a_pos_rnn = self.lstm_model(self.a_pos_embedding)
        # self.a_neg_rnn = self.lstm_model(self.a_neg_embedding)
    def rnn_position_attention(self,fw_cell,bw_cell):
        self.q_rnn = self.lstm_model(fw_cell,bw_cell,self.q_embedding,self.q_len)
        self.a_pos_rnn = self.lstm_model(fw_cell,bw_cell,self.a_embedding,self.a_len)
        self.a_neg_rnn = self.lstm_model(fw_cell,bw_cell,self.a_neg_embedding,self.a_neg_len)
       
        self.q_pos_rnn,self.a_pos_rnn = self.position_attention(self.q_rnn,self.a_pos_rnn,self.a_p,self.a_max_len,self.q_mask,self.a_mask, name="pos_position_attention")
        self.q_neg_rnn,self.a_neg_rnn = self.position_attention(self.q_rnn,self.a_neg_rnn,self.a_neg_p,self.a_neg_max_len,self.q_mask,self.a_neg_mask,name="neg_position_attention")
    def para_initial(self):
        # print(("---------"))
        self.W_qp = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,1],stddev = 0.01,name = 'W_qp'))
        self.U = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,self.hidden_size * 2],stddev = 0.01,name = 'U'))
        self.W_hm = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,self.hidden_size * 2],stddev = 0.01,name = 'W_hm'))
        self.W_qm = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,self.hidden_size * 2],stddev = 0.01,name = 'W_qm'))
        self.W_ms = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,1],stddev = 0.01,name = 'W_ms'))
        self.M_qi = tf.Variable(tf.truncated_normal(shape = [self.hidden_size * 2,self.embedding_size],stddev = 0.01,name = 'M_qi'))

        self.em_hide = tf.Variable(tf.truncated_normal(shape = [self.position_embedding_dim,self.hidden_size * 2],stddev = 0.01, name = 'em_hide'))
    def rnn_att_inner(self,fw_cell,bw_cell):
        self.q_rnn = self.lstm_model(fw_cell,bw_cell,self.q_embedding,self.q_len)
        self.q_pos_rnn,self.a_pos_input = self.inner_attention(self.q_rnn, self.a_embedding, self.q_mask, self.a_mask)
        self.q_neg_rnn,self.a_neg_input = self.inner_attention(self.q_rnn, self.a_neg_embedding, self.q_mask, self.a_neg_mask)
        self.a_pos_rnn_1 = self.lstm_model(fw_cell,bw_cell,self.a_pos_input,self.a_len)
        self.a_neg_rnn_1 = self.lstm_model(fw_cell,bw_cell,self.a_neg_input,self.a_neg_len)       
        self.a_pos_rnn = tf.reduce_mean(self.a_pos_rnn_1,1)
        self.a_neg_rnn = tf.reduce_mean(self.a_neg_rnn_1,1)
    def rnn_att_tra(self,fw_cell,bw_cell):
        self.q_rnn = self.lstm_model(fw_cell,bw_cell,self.q_embedding,self.q_len)
        self.a_pos_rnn = self.lstm_model(fw_cell,bw_cell,self.a_embedding,self.a_len)
        self.a_neg_rnn = self.lstm_model(fw_cell,bw_cell,self.a_neg_embedding,self.a_neg_len)
        self.q_pos_rnn,self.a_pos_rnn = self.rnn_attention(self.q_rnn,self.a_pos_rnn,self.q_mask,self.a_mask)
        self.q_neg_rnn,self.a_neg_rnn = self.rnn_attention(self.q_rnn,self.a_neg_rnn,self.q_mask,self.a_neg_mask)

    def traditional_attention(self,input_left,input_right,q_mask,a_mask):
        input_left_mask = tf.multiply(input_left, tf.expand_dims(tf.cast(q_mask,tf.float32),2))
        Q = tf.reduce_mean(input_left_mask,1)
        a_shape = tf.shape(input_right)
        A = tf.reshape(input_right,[-1,self.hidden_size * 2])
        m_t = tf.nn.tanh(tf.reshape(tf.matmul(A,self.W_hm),[-1,a_shape[1],self.hidden_size*2]) + tf.expand_dims(tf.matmul(Q,self.W_qm),1))
        f_attention = tf.exp(tf.reshape(tf.matmul(tf.reshape(m_t,[-1,self.hidden_size*2]),self.W_ms),[-1,a_shape[1],1]))
        self.f_attention_mask = tf.multiply(f_attention,tf.expand_dims(tf.cast(a_mask,tf.float32),2))
        self.f_attention_norm = tf.divide(self.f_attention_mask,tf.reduce_sum(self.f_attention_mask,1,keep_dims = True))
        self.see = self.f_attention_norm
        a_attention = tf.reduce_sum(tf.multiply(input_right,self.f_attention_norm),1)
        return Q,a_attention


    def position_attention(self,input_left,input_right,input_position,a_len,q_mask,a_mask,name="position_attention"):
        with tf.variable_scope('position_attention' + name):
            input_left_mask = tf.multiply(input_left, tf.expand_dims(tf.cast(q_mask,tf.float32),2))
            Q = tf.reduce_mean(input_left_mask,1,name="Q")
            a_shape = tf.shape(input_right)
            A = tf.reshape(input_right,[-1,self.hidden_size * 2])
            a_position = tf.reshape(input_position,[-1,self.position_embedding_dim])
        
            m_t = tf.nn.tanh(tf.reshape(tf.matmul(A,self.W_hm),[-1,a_shape[1],self.hidden_size * 2]) + \
                tf.reshape(tf.matmul(a_position,self.em_hide),[-1,a_len,self.hidden_size * 2]) + \
                tf.expand_dims(tf.matmul(Q,self.W_qm),1))
            f_attention = tf.exp(tf.reshape(tf.matmul(tf.reshape(m_t,[-1,self.hidden_size * 2]),self.W_ms),[-1,a_len,1]))
            self.f_attention_mask = tf.multiply(f_attention,tf.expand_dims(tf.cast(a_mask,tf.float32),2))
            self.f_attention_norm = tf.divide(self.f_attention_mask,tf.reduce_sum(self.f_attention_mask,1,keep_dims = True))
            self.see = self.f_attention_norm
            a_attention = tf.reduce_sum(tf.multiply(input_right,self.f_attention_norm),1,name="a_attention")
            return Q,a_attention

    def inner_attention(self,input_left,input_right,q_mask,a_mask):        
        input_left_mask = tf.multiply(input_left, tf.expand_dims(tf.cast(q_mask,tf.float32),2))
        self.Q = tf.reduce_mean(input_left_mask,1)
        a_shape = tf.shape(input_right)
        self.M_qi_dropout = tf.nn.dropout(self.M_qi, self.dropout_keep_prob_holder)
        QM = tf.expand_dims(tf.matmul(self.Q,self.M_qi_dropout),1)

        a_value = tf.transpose(tf.matmul(QM,tf.transpose(input_right,[0,2,1])),[0,2,1])
        self.a_t = tf.nn.sigmoid(a_value)
        self.a_t_mask = tf.multiply(self.a_t, tf.expand_dims(tf.cast(  a_mask,tf.float32),2),name = "a_t_mask")
        # self.a_t_norm = tf.divide(self.a_t_mask,tf.reduce_sum(self.a_t_mask,1,keep_dims=True))
        # print((self.a_t_mask))

        self.a_attention = tf.multiply(self.a_t_mask, input_right)
        # self.a_attention = input_right
        # print(("a_att",self.a_attention))
        return self.Q,self.a_attention



    
    def rnn_attention(self,q,a,q_mask,a_mask):

        if self.attention == 'simple':
            q = tf.reduce_mean(q,axis = 1,keep_dims = True)
            alpha = tf.nn.softmax(tf.matmul(a,tf.transpose(q,perm = [0,2,1])),1)
            a_attention = tf.reduce_sum(tf.multiply(a,alpha),axis = 1)
            return tf.squeeze(q,axis = 1),a_attention

        elif self.attention == 'attentive':
            
            q_attention, a_attention = self.attentive_pooling(q,a)
            return q_attention,a_attention
        elif self.attention == 'traditional1':
            q = tf.reduce_mean(q,axis = 1,keep_dims = True)
            first = tf.matmul(tf.reshape(a,[-1,self.hidden_size * 2]),self.U)
            second = tf.reshape(first,[-1,self.max_input_right,self.hidden_size * 2])
            alpha = tf.nn.softmax(tf.matmul(second,tf.transpose(q,perm = [0,2,1])),1)
            a_attention = tf.reduce_sum(tf.multiply(a,alpha),axis = 1)
            return tf.squeeze(q,axis = 1),a_attention
        elif self.attention == 'traditional2':
            q,a_attention = self.traditional_attention(q,a,q_mask,a_mask)
            return q,a_attention
        else:
            return tf.reduce_mean(q,axis = 1),tf.reduce_mean(a,axis = 1)

    def lstm_model(self,fw_cell,bw_cell,embedding_sentence,seq_len):
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,embedding_sentence,sequence_length = seq_len,dtype = tf.float32)
#        print("outputs:===>",outputs) #outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>)))
        #3. concat output
        output_rnn = tf.concat(outputs,axis = 2) #[batch_size,sequence_length,hidden_size*2]
        # self.output_rnn_last = tf.reduce_mean(output_rnn,axis = 1) #[batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
        return output_rnn

    def lstm_cell(self,name):
        with tf.variable_scope('forward' + name):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,1.0)
        with tf.variable_scope('backward' + name):
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)

        if self.dropout_keep_prob_holder is not None:
            self.lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob = self.dropout_keep_prob_holder)
            self.lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob = self.dropout_keep_prob_holder)
        return self.lstm_fw_cell,self.lstm_bw_cell
    def gru_cell(self,name):
        with tf.variable_scope('forward' + name):
            gru_fw_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        with tf.variable_scope('backward' + name):
            gru_bw_cell = tf.contrib.rnn.GRUCell(self.hidden_size)

        if self.dropout_keep_prob_holder is not None:
            self.lstm_fw_cell = rnn.DropoutWrapper(gru_fw_cell,output_keep_prob = self.dropout_keep_prob_holder)
            self.lstm_bw_cell = rnn.DropoutWrapper(gru_bw_cell,output_keep_prob = self.dropout_keep_prob_holder)
        return self.lstm_fw_cell,self.lstm_bw_cell
    def create_loss(self):
        
        with tf.name_scope('score'):
            self.score12 = self.getCosine(self.q_pos_rnn,self.a_pos_rnn ,name="pos_score")
            self.score13 = self.getCosine(self.q_neg_rnn,self.a_neg_rnn,name="neg_score")
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)
    def create_op(self):
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step = self.global_step)


    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled
    def getCosine(self,q,a,name="scores"):
        pooled_flat_1 = tf.nn.dropout(q, self.dropout_keep_prob_holder)
        pooled_flat_2 = tf.nn.dropout(a, self.dropout_keep_prob_holder)
#<<<<<<< HEAD
#        
#        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) 
#        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
#        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) 
#        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name=name) 
#=======
        q_normalize = tf.nn.l2_normalize(q,dim = 1)
        a_normalize = tf.nn.l2_normalize(a,dim = 1)

        score = tf.reduce_sum(tf.multiply(q_normalize,a_normalize),1)
        # pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1)) 
        # pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        # pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1) 
        # score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores") 

        return score
    
    def attentive_pooling(self,input_left,input_right):
       
        # Q = tf.reshape(input_left,[-1,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        # A = tf.reshape(input_right,[-1,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')
        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        Q = input_left
        A = input_right
        first = tf.matmul(tf.reshape(Q,[-1,self.hidden_size * 2]),self.U)
#        print( tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters]))
#        print( self.U)
        second_step = tf.reshape(first,[-1,self.max_input_left,self.hidden_size * 2])
        result = tf.matmul(second_step,tf.transpose(A,perm = [0,2,1]))
        # print( 'result',result)
        G = tf.tanh(result)
        # G = result
        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

        self.attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        self.attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')
        self.see = self.attention_a
        R_q = tf.reshape(tf.matmul(Q,self.attention_q,transpose_a = 1),[-1,self.hidden_size * 2],name = 'R_q')
        R_a = tf.reshape(tf.matmul(self.attention_a,A),[-1,self.hidden_size * 2],name = 'R_a')

        return R_q,R_a
        
    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.total_embedding_dim, 1],
                    padding='SAME',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution_pooling(self):
        print( 'narrow pooling')
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        embeddings = [self.q_pos_embedding,self.q_neg_embedding,self.a_pos_embedding,self.a_neg_embedding]
        self.q_pos_pooling,self.q_neg_pooling,self.a_pos_pooling,self.a_neg_pooling = [self.getFeatureMap(embedding,right = i / 2) for i,embedding in enumerate(embeddings) ]
    
    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_graph(self):
        self.create_placeholder()
        self.create_position()
        self.add_embeddings()

        self.rnn_model_sentence()
        self.create_loss()
        self.create_op()
        self.merged = tf.summary.merge_all()
    
#    @log_time_delta
    def train(self,sess,data):
        feed_dict = {
                self.question:data[0],
                self.answer:data[1],
                self.answer_negative:data[2],
#                self.pos_position:data[3],
#                self.neg_position:data[4],
                # self.q_mask:data[3],
                # self.a_mask:data[4],
                # self.a_neg_mask:data[5],
                self.dropout_keep_prob_holder:self.dropout_keep
            }

        _, summary, step, loss, accuracy,score12, score13, see = sess.run(
                    [self.train_op, self.merged,self.global_step,self.loss, self.accuracy,self.score12,self.score13, self.see],
                    feed_dict)
        return _, summary, step, loss, accuracy,score12, score13, see
    def predict(self,sess,data):
        feed_dict = {
                self.question:data[0],
                self.answer:data[1],
#                self.pos_position:data[2],
                # self.q_mask:data[2],
                # self.a_mask:data[3],
                self.dropout_keep_prob_holder:1.0
            }     
        # print(data[2])       
        score = sess.run( self.score12, feed_dict)       
        return score
    
if __name__ == '__main__':
    from params import Params
    
    from dataset import qa
    params = Params()
    config_file = 'config/qa.ini'    # define dataset in the config
    params.parse_config(config_file)
    reader = qa.setup(params)
    params = qa.process_embedding(reader,params)
    
    cnn = QARNN(params)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])
    q_mask = np.ones((3,33))
    a_mask = np.ones((3,40))
    a_neg_mask = np.ones((3,40))
  

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            # cnn.answer_negative:input_x_3,
            cnn.q_mask:q_mask,
            cnn.a_mask:a_mask,
            cnn.dropout_keep_prob_holder:cnn.dropout_keep
            # cnn.a_neg_mask:a_neg_mask
            # cnn.q_pos_overlap:q_pos_embedding,
            # cnn.q_neg_overlap:q_neg_embedding,
            # cnn.a_pos_overlap:a_pos_embedding,
            # cnn.a_neg_overlap:a_neg_embedding,
            # cnn.q_position:q_position,
            # cnn.a_pos_position:a_pos_position,
            # cnn.a_neg_position:a_neg_position
        }
        question,answer,score = sess.run([cnn.question,cnn.answer,cnn.score12],feed_dict)
        print( question.shape,answer.shape)
        print( score)
