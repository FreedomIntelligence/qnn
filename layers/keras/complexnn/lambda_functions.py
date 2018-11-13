# -*- coding: utf-8 -*-
import keras.backend as K
ALPHA = 0.2
def l2_distance(inputs):
    left = inputs[0]
    right = inputs[1]
    distance = K.sqrt(K.sum((left-right)**2,keepdims = True)+0.01)
    return distance

def cosine_similarity(inputs):
    left = K.l2_normalize(inputs[0],axis = 1)
    right = K.l2_normalize(inputs[1],axis = 1)
    left = K.expand_dims(left)
    right = K.expand_dims(right, axis = 1)
    dot_prod = K.batch_dot(left, right,axes = (1,2))
    dot_prod = K.reshape(dot_prod, shape = (-1,1))
#    print(dot_prod.shape)
    return dot_prod

def triplet_hinge_loss(inputs):
    anchor = inputs[0]
    positive = inputs[1]
    negative = inputs[2]

    pos_dist = K.sum((anchor-positive)**2, keepdims = True)
    neg_dist = K.sum((anchor-negative)**2, keepdims = True)

    basic_loss = pos_dist-neg_dist+ALPHA
    loss = K.mean(K.maximum(basic_loss, 0.0),keepdims = True)

    return loss


#a = Input(shape=(10,))
#b = Input(shape=(10,))
#
#output = Lambda(cosine_similarity)([a,b])
#
#model = Model(inputs=[a, b], outputs=output)
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#
## Generating Some data
#a_s = np.ones((5,10))
#b_s = np.ones((5,10))
#print(model.predict([a_s,b_s]))
