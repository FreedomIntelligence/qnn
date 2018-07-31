# -*- coding: utf-8 -*-
from params import Params
import models
import dataset
params = Params()
params.parse_config('config/config.ini')
params.network_type = "complex"
params.network_type = "qdnn"
reader=dataset.setup(params)
model = models.setup(params).getModel()

model.compile(loss = params.loss,
          optimizer = params.optimizer,
          metrics=['accuracy'])

model.summary()
weights = model.get_weights()




(train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()

history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

evaluation = model.evaluate(x = test_x, y = test_y)