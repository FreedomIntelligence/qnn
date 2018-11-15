# -*- coding: utf-8 -*-
import codecs
class Dictionary(dict):
    def __init__(self, start_feature_id = 1,dict_path=None):
        self.fid = start_feature_id
        if dict_path is not None:
            self.load_from_file(dict_path)

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))
                
    def load_from_file(self,dict_path):

        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self[token] = len(self)

if __name__ == '__main__':
    dict_path =     "D:/dataset/bert/uncased_L-12_H-768_A-12/vocab.txt"
    dic = Dictionary(dict_path=dict_path)

    print(dic)