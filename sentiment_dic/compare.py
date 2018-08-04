import pandas as pd
from pandas import DataFrame
df_1 = pd.read_table('true_word_sentiment.txt', delim_whitespace=True, names=('word', 'sentiment_1'), dtype={'word': str, 'B': float }, encoding = 'gbk')

df_2 = pd.read_table('../word_sentiment.txt', delim_whitespace=True, names=('word', 'sentiment_2'), dtype={'word': str, 'B': float }, encoding = 'gbk')

df_3 = df_1.merge(df_2)

df_3["sentiment_2"]=df_3["sentiment_2"].apply(lambda x: 0  if x-0.5 <1e-5 else 1)
print(df_3)
yes= sum(df_3["sentiment_1"]==df_3["sentiment_2"])
print(yes *1.0/ len(df_3))
# print(df_3)
