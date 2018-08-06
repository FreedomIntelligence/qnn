import os
import io,re,codecs
import numpy as np
import configparser
import argparse
class Params(object):
    def __init__(self):
        pass
    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        for key,value in config_common.items():
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else :
                    value= float(value)

            self.__dict__.__setitem__(key,value)            

    def export_to_config(self, config_file_path):
        config = configparser.ConfigParser()
        config['COMMON'] = {}
        config_common = config['COMMON']
        for k,v in self.__dict__.items():        
            if not k == 'lookup_table':    
                config_common[k] = str(v)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    def parseArgs(self):
        #required arguments:
        parser = argparse.ArgumentParser(description='running the complex embedding network')
        parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')
        args = parser.parse_args()
        self.parse_config(args.config_file_path)
    
    def setup(self,parameters):
        for k, v in parameters:
            self.__dict__.__setitem__(k,v)
    def get_parameter_list(self):
        info=[]
        for k, v in self.__dict__.items():
            info.append("%s -> %s"%(k,str(v)))
        return info
    def to_string(self):
        return " ".join(self.get_parameter_list())
    def save(self,path):
        with codecs.open(path+"/config.ini","w",encoding="utf-8") as f:
            f.write("\n".join(self.get_parameter_list()))
        


