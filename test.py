# -*- coding: utf-8 -*-
import keras
from keras.layers import Input, Dense, Activation
import numpy as np
from keras import regularizers
from keras.models import Model
import sys

from params import Params
import models
import dataset
from save import save_experiment
import keras.backend as K
import units



params = Params()
config_file = 'config/local.ini'    # define dataset in the config
params.parse_config(config_file)
reader = dataset.setup(params)
params = dataset.process_embedding(reader,params)
qdnn = models.setup(params)
model = qdnn.getModel()

model.compile(loss = params.loss,
            optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
            metrics=['accuracy'])
model.summary()
(train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()

history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

evaluation = model.evaluate(x = val_x, y = val_y)



# x_input = np.asarray([b])
# y = model.predict(x_input)
# print(y)


