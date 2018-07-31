import os
import io
import logging
import numpy as np
import configparser
import argparse
class Params(object):
    def __init__(self, datasets_dir = None, dataset_name = None, wordvec_initialization ='random', wordvec_path = None, eval_dir = None, network_type = 'complex_mixture',embedding_trainable = True, loss = 'binary_crossentropy', optimizer = 'rmsprop', initial_mode = 'he',batch_size = 16, epochs= 4, dropout_rate = 0.5, activation = 'sigmoid'):
        self.datasets_dir = datasets_dir
        self.dataset_name = dataset_name
        self.wordvec_initialization = wordvec_initialization
        self.wordvec_path = wordvec_path
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size =batch_size
        self.epochs = epochs
        self.eval_dir = eval_dir
        self.network_type = network_type
        self.embedding_trainable = embedding_trainable
        self.dropout_rate = dropout_rate
        self.init_mode = initial_mode
        self.activation = activation

    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']

        if 'datasets_dir' in config_common:
            self.datasets_dir = config_common['datasets_dir']

        if 'dataset_name' in config_common:
            self.dataset_name = config_common['dataset_name']

        if 'wordvec_initialization' in config_common:
            self.wordvec_initialization = config_common['wordvec_initialization']

        if 'wordvec_path' in config_common:
            self.wordvec_path = config_common['wordvec_path']

        if 'loss' in config_common:
            self.loss = config_common['loss']

        if 'optimizer' in config_common:
            self.optimizer = config_common['optimizer']

        if 'batch_size' in config_common:
            self.batch_size = int(config_common['batch_size'])

        if 'epochs' in config_common:
            self.epochs = int(config_common['epochs'])

        if 'eval_dir' in config_common:
            self.eval_dir = config_common['eval_dir']

        if 'network_type' in config_common:
            self.network_type = config_common['network_type']

        if 'embedding_trainable' in config_common:
            self.embedding_trainable = bool(config_common['embedding_trainable'])
        if 'dropout_rate' in config_common:
            self.dropout_rate = float(config_common['dropout_rate'])

        if 'initial_mode' in config_common:
            self.initial_mode = config_common['initial_mode']

        if 'activation' in config_common:
            self.activation = config_common['activation']

    def export_to_config(self, config_file_path):
        config = configparser.ConfigParser()
        config['COMMON'] = {}
        config_common = config['COMMON']
        config_common['datasets_dir'] = self.datasets_dir
        config_common['dataset_name'] = self.dataset_name
        config_common['wordvec_initialization'] = self.wordvec_initialization
        config_common['wordvec_path'] = self.wordvec_path
        config_common['loss'] = self.loss
        config_common['optimizer'] = self.optimizer
        config_common['batch_size'] = str(self.batch_size)
        config_common['epochs'] = str(self.epochs)
        config_common['eval_dir'] = str(self.eval_dir)
        config_common['network_type'] = self.network_type
        config_common['embedding_trainable'] = str(self.embedding_trainable)
        config_common['dropout_rate'] = str(self.dropout_rate)
        config_common['initial_mode'] = self.initial_mode
        config_common['activation'] = self.activation

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    def parseArgs(self):
        #required arguments:
        parser = argparse.ArgumentParser(description='running the complex embedding network')
        parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')
        args = parser.parse_args()
        self.parse_config(args.config_file_path)


