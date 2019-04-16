# -*- coding: utf-8 -*-

from keras.models import load_model
from params import Params
from dataset import qa
from layers import *
from layers.loss import *
from layers.loss.metrics import precision_batch
from tools.units import to_array
from tools.evaluation import matching_score
params = Params()
params.parse_config("config/qalocal_pair_trec.ini")
reader = qa.setup(params)

model = load_model('temp/best.h5',custom_objects={'NGram': NGram,"L2Normalization":L2Normalization,"L2Norm":L2Norm,"ComplexMeasurement":ComplexMeasurement,"ComplexMultiply":ComplexMultiply,"ComplexMixture":ComplexMixture,"Concatenation":Concatenation,"Cosine":Cosine,"MarginLoss":MarginLoss,"identity_loss":identity_loss,"precision_batch":precision_batch})

test_data = reader.getTest(iterable = False, mode = 'test')
test_data.append(test_data[0])
test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
y_pred = model.predict(x = test_data) 
score = matching_score(y_pred, params.onehot, params.match_type)    
test_metric = reader.evaluate(score, mode = "test")
print(test_metric)