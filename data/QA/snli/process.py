import os
from codecs import open
import pandas as pd

l = {"entailment":1 , "contradiction": 0}
for extension in ['train','test','dev']:
    s1_file = 's1.{}'.format(extension)
    s2_file = 's2.{}'.format(extension)
    label_file = 'labels.{}'.format(extension)
    
    left = pd.read_csv(s1_file,sep = "\t",names = ["left"])
    right = pd.read_csv(s2_file,sep = "\t",names = ["right"])
    label = pd.read_csv(label_file,sep = "\t",names = ["label"])

#    .drop().replace(l)
    table = pd.concat([left,right,label],axis = 1)
    table= table[table.label!='neutral'].replace(l)
    table.to_csv('{}.txt'.format(extension), sep = "\t", header = None)
#    print(s1_file)
    
    
#for filename in ["train.txt","test.txt"]:
#
#	fin=filename
#	df= pd.read_csv(fin,sep="\t",encoding="utf-8",quoting=3)
#	df[["#1 String", "#2 String","Quality"]].to_csv(filename,index=False,header=None,encoding="utf-8",sep="\t")