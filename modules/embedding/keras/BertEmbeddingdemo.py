import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute
from keras.models import Model, Input, model_from_json, load_model

if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    checkpoint_path="D:/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt" #chinese_L-12_H-768_A-12
    config_path =   "D:/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json" #chinese_L-12_H-768_A-12
    dict_path =     "D:/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt" #chinese_L-12_H-768_A-12
else:
    config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
model.summary(line_length=120)

tokens = ['[CLS]', '语', '言', '模', '型', '[SEP]']
#dense_output = Dense(100, activation='relu')(model.output)
#head_model = Model(input = model.input, output = dense_output)
#
##compile the model
##head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
#head_model.summary()


token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])

print(token_input[0][:len(tokens)])

predicts = model.predict([token_input, seg_input])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])