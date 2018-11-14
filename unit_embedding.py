# -*- coding: utf-8 -*-

from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
import codecs
import numpy as np
from layers.keras.complexnn.l2_normalization import L2normalization
encoding_dim = 100
mnist_config = False#False
if mnist_config:
    input_dim = 784
    
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print (x_train.shape)
    print (x_test.shape)

else:
    input_dim = 300

input_img = Input(shape=(input_dim,))
# add a Dense layer with a L1 activity regularizer
import keras.backend as K
if mnist_config:
    encoded = Dense(encoding_dim,activity_regularizer=regularizers.l1(10e-7))(input_img) #,  
    normed_encode=L2normalization()(encoded)
#    normed_encode=encoded
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
else:
    encoded = Dense(encoding_dim)(input_img) #, 
    normed_encode = L2normalization()(encoded)
    decoded = Dense(input_dim)(normed_encode)
    
    
    
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, normed_encode)
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
if mnist_config:
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
else:
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')





if mnist_config:
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))   
else:
#    import itertools
#    np.array([i[0] for i in itertools.islice(, 1000)])
    def generate_from_file(file_name,batch_size=256):
        while 1:
            with codecs.open(file_name, 'r',encoding='utf-8') as f:
                batch_image=[];
                for line in f:
                    word, vec = line.split(' ', 1)
                    vector = np.fromstring(vec, sep=' ')
                    batch_image.append(vector)
                    if len(batch_image) ==batch_size:
                        yield (np.array(batch_image), np.array(batch_image))
                        batch_image=[]
    x_test =  np.array(next(generate_from_file("glove/glove.6B.300d.txt",256*8))[0])
#    autoencoder.fit_generator(generate_from_file("glove/glove.6B.300d.txt"),
#                    samples_per_epoch=10000, nb_epoch=10,
#                    validation_data=(x_test, x_test))
    def getdata(file_name):
        with codecs.open(file_name, 'r',encoding='utf-8') as f:
            batch_image=[];
            for line in f:
                word, vec = line.split(' ', 1)
                vector = np.fromstring(vec, sep=' ')
                batch_image.append(vector)
        return np.array(batch_image)
    
    batchse = getdata("glove/glove.6B.300d.txt")
    autoencoder.fit(batchse, batchse,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    x_test= x_test[:10]

    
decoded_imgs = autoencoder.predict(x_test)

    

#import matplotlib.pyplot as plt
#
#n = 10  # how many digits we will display
#plt.figure(figsize=(20, 4))
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i + 1)
#    plt.imshow(x_test[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    # display reconstruction
#    ax = plt.subplot(2, n, i + 1 + n)
#    plt.imshow(decoded_imgs[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()


def generate_word_vector(file_name,batch_size=256):

    with codecs.open(file_name, 'r',encoding='utf-8') as f:
        batch_image=[]
        words=[]
        for line in f:
            word, vec = line.split(' ', 1)
            vector = np.fromstring(vec, sep=' ')
            batch_image.append(vector)
            words.append(word)
            if len(batch_image) ==batch_size:
                yield words,np.array(batch_image)
                batch_image=[]
                words=[]
output_file="glove/normalized_vectors.txt"
with codecs.open(output_file, 'w',encoding='utf-8') as outf:
    for words,vectors in generate_word_vector("glove/glove.6B.300d.txt"):
        new_vectors = encoder.predict(vectors)
        for index,word in enumerate(words):
            vector_str = " ".join([str(i) for i in new_vectors[index]])
            outf.write("%s %s"%(word,vector_str))
        
    

    
