import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
import sys
import os
import keras.backend as K
import math


class ComplexSuperposition(Layer):

    def __init__(self, average_weights = False,**kwargs):
        # self.output_dim = output_dim
        self. average_weights = average_weights
        super(ComplexSuperposition, self).__init__(**kwargs)

    def get_config(self):
        config = {'average_weights': self.average_weights}
        base_config = super(ComplexSuperposition, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(input_shape) != 3 and len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')

        super(ComplexSuperposition, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(inputs) != 3 and len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2/3 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')


        # if len(inputs) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on only 1 input.'
        #                      'Got ' + str(len(input)) + ' inputs.')
        input_real = inputs[0]
        input_imag = inputs[1]
        
        ndims = len(inputs[0].shape)
        if self.average_weights:
            output_r = K.mean(input_real,axis = ndims-2, keepdims = False)
            output_i = K.mean(input_imag,axis = ndims-2, keepdims = False)
        else:
            #For embedding layer inputs[2] is (None, embedding_dim,1)
            #For test inputs[2] is (None, embedding_dim)
            if len(inputs[2].shape) == ndims-1:
                weight = K.expand_dims(inputs[2])
            else:
                weight = inputs[2]

            weight = K.repeat_elements(weight, input_real.shape[-1], axis = ndims-1)

            output_real = input_real*weight #shape: (None, 300, 300)
            output_real = K.sum(output_real, axis = ndims-2)
            output_imag = input_imag*weight
            output_imag = K.sum(output_imag, axis = ndims-2)
        
        
        output_real_transpose = K.expand_dims(output_real,axis = ndims-2)
        output_imag_transpose = K.expand_dims(output_imag,axis = ndims-2)



#        output_real_transpose = K.permute_dimensions(output_real, (0,2,1))
#        output_imag_transpose = K.permute_dimensions(output_imag, (0,2,1))
        
        output_real = K.expand_dims(output_real)
        output_imag = K.expand_dims(output_imag)
        

        print(output_real.shape)
        print(output_real_transpose.shape)
        # print(output_imag.shape)

        output_r = K.batch_dot(output_real,output_real_transpose, axes = [ndims-1,ndims]) + K.batch_dot(output_imag,output_imag_transpose, axes = [ndims-1,ndims])
        output_i = K.batch_dot(output_imag,output_real_transpose, axes = [ndims-1,ndims]) - K.batch_dot(output_real,output_imag_transpose, axes = [ndims-1,ndims])

        return [output_r, output_i]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        
        one_input_shape = list(input_shape[0])
        one_output_shape = []
        for i in range(len(one_input_shape)):
            if not i== len(one_input_shape)-2:
                one_output_shape.append(one_input_shape[i])
        one_output_shape.append(one_output_shape[-1])
#        
#        one_input_shape = list(input_shape[0])
#        one_output_shape = [one_input_shape[0], one_input_shape[2], one_input_shape[2]]
        return [tuple(one_output_shape), tuple(one_output_shape)]



def main():
    input_2 = Input(shape=(2,3,5), dtype='float')
    input_1 = Input(shape=(2,3,5), dtype='float')
    weights = Input(shape = (2,3,),dtype= 'float')
    [output_1, output_2] = ComplexSuperposition(average_weights = False)([input_1, input_2, weights])


    model = Model([input_1, input_2, weights], [output_1, output_2])
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    x = np.random.random((4,2,3,5))
    x_2 = np.random.random((4,2,3,5))

    weights = np.random.random((4,2,3))
    output = model.predict([x,x_2, weights])
    print(output[0].shape)


    # rng = numpy.random.RandomState(123)

    # Warning: the full dataset is over 40GB. Make sure you have enough RAM!
    # This can take a few minutes to load
    # if in_memory:
    #     print('.. loading train data')
    #     dataset = MusicNet(local_data, complex_=complex_, fourier=fourier,
    #                        stft=stft, rng=rng, fast_load=fast_load)
    #     dataset.load()
    #     print('.. train data loaded')
    #     Xvalid, Yvalid = dataset.eval_set('valid')
    #     Xtest, Ytest = dataset.eval_set('test')
    # else:
    #     raise ValueError

    # print(".. building model")
    # # model = get_shallow_convnet(window_size=4096, channels=2, output_size=84)
    # model = one_hidden_layer_complex_nn(input_size = 300, output_size = 2)
    # model.summary()
    # print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))


    # # x =
    # x = np.random.random((1,300))
    # y = to_categorical(np.random.randint(2, size=(1, 1)), num_classes=2)


    # for i in range(700):
    #     model.fit(x,y)

    # print(y)
    # print(model.predict(x))
    # if in_memory:
    #     pass
    #     # do nothing
    # else:
    #     raise ValueError

    # logger = mimir.Logger(
    #     filename='models/log_{}.jsonl.gz'.format(model_name))

    # it = dataset.train_iterator()

    # callbacks = [Validation(Xvalid, Yvalid, 'valid', logger),
    #              Validation(Xtest, Ytest, 'test', logger),
    #              SaveLastModel("./models/", 1, name=model),
    #              Performance(logger),
    #              LearningRateScheduler(schedule)]

    # print('.. start training')
    # model.fit_generator(
    #     it, steps_per_epoch=1000, epochs=epochs,
    #     callbacks=callbacks, workers=1)

if __name__ == '__main__':
    main()
