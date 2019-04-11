# -*- coding: utf-8 -*-
from keras.layers import Layer

class Index(Layer):

    def __init__(self, index = 0, **kwargs):
        self.index = index
        super(Index, self).__init__(**kwargs)

    def get_config(self):
        config = {'index': self.index}
        base_config = super(Index, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        super(Index, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        output = inputs[:,self.index,:]


        return(output)

    def compute_output_shape(self, input_shape):
        output_shape = [None, input_shape[-1]]
        return([tuple(output_shape)])

#def main():
#
#    input_1 = Input(shape=(5,5), dtype='float')
#    output = Index(3)(input_1)
#
#
#    model = Model(input_1, output)
#    model.compile(loss='binary_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])
#    model.summary()
#
#    x = np.random.random((3,5,5))
#    y = model.predict(x)
#    print(y)
#    print(x[:,3,:])
#
#
#
#if __name__ == '__main__':
#    main()

