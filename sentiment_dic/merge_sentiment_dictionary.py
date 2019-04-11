import os
import codecs
import string
pos_file_name = 'positive-words.txt'
pos_file = codecs.open(pos_file_name)
out_file_name = 'word_sentiment.txt'
out_file = codecs.open(out_file_name,'w')
for line in pos_file.readlines():
    if line.strip() == '':
        continue
    if not line.startswith(';'):
        out_file.write(line.strip()+ ' '+ str(1.0)+'\n')

neg_file_name = 'negative-words.txt'
neg_file = codecs.open(neg_file_name)
for line in neg_file.readlines():
    if line.strip() == '':
        continue
    if not line.startswith(';'):
        out_file.write(line.strip()+ ' '+ str(0.0)+'\n')
