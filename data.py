from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
import cmath
import random
import math
import os
from keras.preprocessing.sequence import pad_sequences

def load_complex_embedding(embedding_dir):
    word2id = np.load(os.path.join(embedding_dir,'word2id.npy')).item()
    phase_embedding = np.load(os.path.join(embedding_dir,'phase_embedding.npy'))
    amplitude_embedding = np.load(os.path.join(embedding_dir,'amplitude_embedding.npy'))
    complex_embedding_params = {'word2id':word2id, 'phase_embedding': phase_embedding, 'amplitude_embedding': amplitude_embedding}
    print(phase_embedding.shape)
    print(amplitude_embedding.shape)
    print(len(word2id))
    return complex_embedding_params


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    # words['<s>'] = 1e9 + 4
    # words['</s>'] = 1e9 + 3
    # words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i+1

    return id2word, word2id



def form_matrix(file_name):
    word_list = []
    ll = []
    with io.open(file_name, 'r',encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            ll.append(np.fromstring(vec, sep=' '))
            word_list.append(word)
        matrix = np.asarray(ll)
    return matrix, word_list



def orthonormalized_word_embeddings(word_embeddings_file):

    matrix, word_list = form_matrix(word_embeddings_file)
    print('Initial matrix constructed!')
    matrix_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    matrix = matrix.astype(np.float)
    matrix_sum = np.sqrt(np.sum(np.square(matrix), axis=1))
    for i in range(np.shape(matrix)[0]):
        matrix_norm[i] = matrix[i]/matrix_sum[i]
    print('Matrix normalized')

    ##q - basis vectors(num_words x dimension).
    ##r - coefficients of each word in the basis(dimension x num_words)
    q, r = np.linalg.qr(np.transpose(matrix_norm), mode = 'complete')
    print('qr factorization completed. Matrix orthogonalized!')

    ## Dot product of king and prince vectors same as in the original embeddings (0.76823)
    # king = word_list.index('king')
    # prince = word_list.index('prince')
    # print (np.dot(r[:, king], r[:, prince]))
    return r, word_list


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id=None, orthonormalized=True):
    if orthonormalized:
        coefficients_matrix, word_list = orthonormalized_word_embeddings(path_to_vec)
    else:
        matrix, word_list = form_matrix(path_to_vec)
        coefficients_matrix = np.transpose(matrix)
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        if word2id == None:
            print('program goes here!')
            for word in word_list:
                word_vec[word] = coefficients_matrix[:, word_list.index(word)]
        else:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word2id:
                    word_vec[word] = coefficients_matrix[:, word_list.index(word)]

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

# Set word phases
# Currently only using random phase
def set_wordphase(word2id):
    word2phase = {}
    for word in word2id.keys():
        word2phase[word] = random.random()*2*math.pi

    return word2phase

def get_index_batch(embedding_params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    word2id = embedding_params['word2id']
    for sent in batch:
        sentvec = []
        for word in sent:
            if word in word2id:
                assert word2id[word] > 0
                sentvec.append(word2id[word])

        if not sentvec:
            vec = np.zeros(embedding_params['wvec_dim'])
            sentvec.append(vec)

        # word_count = len(sentvec)
        # sentvec = np.mean(sentvec, 0)*math.sqrt(word_count)
        embeddings.append(sentvec)

    # embeddings = np.vstack(embeddings)
    return embeddings


def get_vector_batch(embedding_params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in embedding_params['word_vec']:
                wordvec = embedding_params['word_vec'][word]
                if word in embedding_params['word_complex_phase']:
                    complex_phase = embedding_params['word_complex_phase'][word]
                    wordvec = [x * cmath.exp(1j*complex_phase) for x in wordvec]
                sentvec.append(wordvec)
        if not sentvec:
            vec = np.zeros(embedding_params['wvec_dim'])
            sentvec.append(vec)
        word_count = len(sentvec)
        sentvec = np.mean(sentvec, 0)*math.sqrt(word_count)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

def get_lookup_table(embedding_params):
    id2word = embedding_params['id2word']
    word_vec = embedding_params['word_vec']
    lookup_table = []

    # Index 0 corresponds to nothing
    lookup_table.append([0]* embedding_params['wvec_dim'])
    for i in range(0, len(id2word)):
        word = id2word[i]
        wvec = [0]* embedding_params['wvec_dim']
        if word in word_vec:
            wvec = word_vec[word]
        # print(wvec)
        lookup_table.append(wvec)

    lookup_table = np.asarray(lookup_table)
    return(lookup_table)


def batch_gen(data, max_sequence_length):
    sentences = data['X']
    labels = data['y']
    # print(labels)
    for batch, label in zip(sentences, labels):
        padded_batch = pad_sequences(batch, maxlen=max_sequence_length, dtype='int32',
        padding='post', truncating='post', value=0.)
        yield np.asarray(padded_batch), np.asarray(label)

def data_gen(data, max_sequence_length):
    sentences = data['X']
    labels = data['y']
    # print(labels)
    # batch_list = []

    # for batch, label in zip(sentences, labels):
    #     padded_batch = pad_sequences(batch, maxlen=max_sequence_length, dtype='int32',
    #     padding='post', truncating='post', value=0.)
    #     batch_list=padded_batch
    padded_sentences = pad_sequences(sentences[0], maxlen=max_sequence_length, dtype='int32',padding='post', truncating='post', value=0.)

    return np.asarray(padded_sentences), np.transpose(np.asarray(labels))

def main():
    complex_embedding_dir = 'eval/eval_CR/embedding'
    load_complex_embedding(complex_embedding_dir)

if __name__ == '__main__':
        main()

