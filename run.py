# -*- coding: utf-8 -*-
from params import Params
import models
import dataset
from keras import optimizers

params = Params()
params.parse_config('config/qdnn.ini')
params.network_type = "real"
params.network_type = "complex"
params.network_type = "qdnn"
reader=dataset.setup(params)
qdnn = models.setup(params)
model = qdnn.getModel()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rms = optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06)
model.compile(loss = params.loss,
#          optimizer = sgd,
          optimizer = rms,
          metrics=['accuracy'])

model.summary()
weights = model.get_weights()



(train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()

history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

evaluation = model.evaluate(x = val_x, y = val_y)