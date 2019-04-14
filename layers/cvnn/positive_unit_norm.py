from keras.constraints import Constraint
import keras.backend as K
class PositiveUnitNorm (Constraint):
    def __init__(self, axis = 0):
        self.axis = axis


    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        new_w = w / (K.epsilon() + K.sqrt(K.sum(K.square(w),
                                               axis=self.axis,
                                               keepdims=True)))
        return new_w

    def get_config(self):
        return {'axis': self.axis}
