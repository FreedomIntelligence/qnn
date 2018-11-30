
# -*- coding: utf-8 -*-

from layers.pytorch.complexnn import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.logger import Logger
logger = Logger()     
import os



def myzip(train_x,train_x_mask):
    assert train_x.shape == train_x_mask.shape
    results=[]
    for i in range(len(train_x)):
        results.append((train_x[i],train_x_mask[i]))
    return results
'''

def run(params):
#    if params.bert_enabled == True:
#        params.max_sequence_length = 512
#        params.reader.max_sequence_length = 512
    evaluation=[]
#    params=dataset.classification.process_embedding(reader,params)    
    qdnn = models.setup(params)
    model = qdnn.getModel()
    model.summary()
    if hasattr(loss.pairwise_loss, params.loss): 
        loss_func = getattr(loss.pairwise_loss, params.loss)
    else:
        loss_func = params.loss
    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr)
#    
    
#    test_data = [to_array(i,params.max_sequence_length) for i in test_data]
    if hasattr(loss.pairwise_loss, params.metric_type):
        metric_func = getattr(loss.pairwise_loss, params.metric_type)
    else:
        metric_func = params.metric_type
    
    model.compile(loss = loss_func, #""
                      optimizer = optimizer,
                      metrics=[metric_func])
    # pairwise:
    # loss = identity_loss
    # metric = precision_batch

    # pointwise:
    # loss = categorical_hinge or mean_squared_error
    # metric = acc or mean_squared_error
    
    # classification:
    # loss = mean_squared_error
    # matrix = acc
      
    if params.dataset_type == 'qa':
        test_data = params.reader.get_test(iterable = False)
#        from models.match import keras as models   
        for i in range(params.epochs):
            model.fit_generator(params.reader.batch_gen(params.reader.get_train(iterable = True)),epochs = 1,steps_per_epoch=int(len(reader.datas["train"])/reader.batch_size),verbose = True)        
            y_pred = model.predict(x = test_data) 
            score = batch_softmax_with_first_item(y_pred)[:,1]  if params.onehot else y_pred
                
            metric = params.reader.evaluate(score, mode = "test")
            evaluation.append(metric)
            print(metric)
            logger.info(metric)
        df=pd.DataFrame(evaluation,columns=["map","mrr","p1"]) 

            
    elif params.dataset_type == 'classification':
#        from models import representation as models   
        
        
    #    model.summary()    
#        train_data = params.reader.get_train(iterable = False)
#        test_data = params.reader.get_test(iterable = False)
#        val_data =params.reader.get_val(iterable = False)
#    #    (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
#        train_x, train_y = train_data
#        test_x, test_y = test_data
#        val_x, val_y = val_data
        train_x,train_y = params.reader.get_train(iterable = False)
        test_x, test_y = params.reader.get_test(iterable = False)
        val_x,val_y = params.reader.get_val(iterable = False)
        
        history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))
        
        metric = model.evaluate(x = val_x, y = val_y)   # !!!!!! change the order to val and test
        
        evaluation.append(metric)
        logger.info(metric)
        print(history)
        print(metric)

        df=pd.DataFrame(evaluation,columns=["map","mrr","p1"])  
        
    logger.info("\n".join([params.to_string(),"score: "+str(df.max().to_dict())]))

    K.clear_session()
'''

if __name__=="__main__":
        
#    embedding_matrix = torch.randn(512, 1000)
#    num_measurements = 10
#    model = QDNN(embedding_matrix, num_measurements)
#    
#    input_seq = torch.randint(0,1000, (5,20,)).long()
#    y_pred = model(input_seq)
    
    
#    parser = argparse.ArgumentParser(description='running the complex embedding network')
#    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
#    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
#    args = parser.parse_args()      
#    params = Params()
#    config_file = 'config/config_qdnn.ini'    # define dataset in the config
#    params.parse_config(config_file)    
    class QDNN(nn.Module):
        def __init__(self, embedding_matrix, num_measurements):
            """
            max_sequence_len: input sentence length
            embedding_dim: input dimension
            num_measurements: number of measurement units, also the output dimension

            """
            super(QDNN, self).__init__()
            self.vocab_dim = embedding_matrix.shape[0]
            self.embedding_dim = embedding_matrix.shape[1]
            self.phase_embedding_layer = PhaseEmbedding(self.vocab_dim, self.embedding_dim)
            self.amplitude_embedding_layer = AmplitudeEmbedding(embedding_matrix, random_init = False)
            self.l2_norm = L2Norm(dim= -1, keep_dims = False)
            self.l2_normalization = L2Normalization(dim = -1)
            self.activation = F.softmax
            self.complex_multiply = ComplexMultiply()
            self.mixture = ComplexMixture(average_weights = False)
            self.measurement = ComplexMeasurement(50, units = num_measurements)
#            self.output = ComplexMeasurement(units = self.opt.measurement_size)([self.sentence_embedding_real, self.sentence_embedding_imag])
        def forward(self, input_seq):
            """
            In the forward function we accept a Variable of input data and we must 
            return a Variable of output data. We can use Modules defined in the 
            constructor as well as arbitrary operators on Variables.
            """
            phase_embedding = self.phase_embedding_layer(input_seq)
            amplitude_embedding = self.amplitude_embedding_layer(input_seq)
            weights = self.l2_norm(amplitude_embedding)
            amplitude_embedding = self.l2_normalization(amplitude_embedding)
            weights = self.activation(weights, dim=-1)
            [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
            [sentence_embedding_real, sentence_embedding_imag] = self.mixture([seq_embedding_real, seq_embedding_imag,weights])
            output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
            
            return output
 # import argparse
#    parser = argparse.ArgumentParser(description='running the complex embedding network')
#    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
#    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
#    args = parser.parse_args()      
#    params = Params()
#    config_file = 'config/config_qdnn.ini'    # define dataset in the config
#    params.parse_config(config_file)    
#    
#    reader = dataset.setup(params)y_pred
#    params.reader = reader
    N, D_in, H, D_out = 32, 100, 50, 10
    model = QDNN(torch.randn(50, 50), 3)
    # Construct our model by instantiating the class defined abov
    losses = []
    loss_function = nn.MSELoss()

    # Forward pass: Compute predicted y by passing x to the model
#    y_pred = model(x)   # dim: 32 x 10
    # pick an SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
#    total_loss = 0
    for epoch in range(1):
       
    
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        x_input = torch.tensor([[1,2,3],[2,45,8]],dtype = torch.long)
    
        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        optimizer.zero_grad()
        y_pred = model(x_input)
#>>>>>>> b71565ab6aab4ec59034f3bbc84af1d088171336
#    
#    reader = dataset.setup(params)
#    params.reader = reader
#    

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(y_pred[0], torch.tensor(torch.randn(2,3), dtype=torch.float))  
        loss.backward()
        optimizer.step()
        total_loss = loss.item()
        losses.append(total_loss)
    print(losses)

    
    
 # import argparse

    
#    N, D_in, H, D_out = 32, 100, 50, 10
#    x = Variable(torch.randn(N, D_in))  # dim: 32 x 100
#
#    # Construct our model by instantiating the class defined abov
#    losses = []
#    loss_function = nn.MSELoss()
#
#    # Forward pass: Compute predicted y by passing x to the model
##    y_pred = model(x)   # dim: 32 x 10
#    # pick an SGD optimizer
#    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
##    total_loss = 0
#    for epoch in range(100):
#       
#    
#        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
#        # into integer indices and wrap them in tensors)
#        x_input = torch.tensor(torch.randn(N, D_in),dtype = torch.float)
#    
#        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
#        # new instance, you need to zero out the gradients from the old
#        # instance
#        model.zero_grad()
#    
#        # Step 3. Run the forward pass, getting log probabilities over next
#        # words
#        y_pred = model(x_input)
#    
#        # Step 4. Compute your loss function. (Again, Torch wants the target
#        # word wrapped in a tensor)
#        loss = loss_function(y_pred, torch.tensor(torch.randn(N,D_out), dtype=torch.float))
#    
#        # Step 5. Do the backward pass and update the gradient
#        loss.backward()
#        optimizer.step()
#
#    # Get the Python number from a 1-element Tensor by calling tensor.item()
#        total_loss = loss.item()
#        losses.append(total_loss)
#    print(losses)  # The loss decreased every iteration over the training data!
##    run(params)

