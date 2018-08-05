# -*- coding: utf-8 -*-
from params import Params
import models
import dataset
import codecs
from keras import optimizers
from save import save_experiment

params = Params()
config_file = 'config/qdnn.ini'
params.parse_config(config_file)
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


# word2id = reader.embedding_params['word2id']
# file_name = 'sentiment_dic/sentiment_dic.txt'
# pretrain_x = []
# pretrain_y = []
# with codecs.open(file_name, 'r') as f:
#     for line in f:
#         word, polarity = line.split()
#         if word in word2id:
#             word_id = word2id[word]
#             pretrain_x.append([word_id]* reader.max_sentence_length)
#             pretrain_y.append(int(float(polarity)))

# pretrain_x = np.asarray(pretrain_x)
# pretrain_y = to_categorical(pretrain_y)

(train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()

#model.fit(x=pretrain_x, y = pretrain_y, batch_size = params.batch_size, epochs= 3,validation_data= (test_x, test_y))

history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

evaluation = model.evaluate(x = val_x, y = val_y)
#save_experiment(model, params, evaluation, history, reader, config_file)
