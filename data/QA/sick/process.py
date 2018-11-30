from codecs import open
import os
import pandas as pd
files = [("SICK_train.txt","train.txt"),("SICK_test_annotated.txt","test.txt")]
for old,new in files:
	df = pd.read_csv(old,sep="\t")
	# sentence_A	sentence_B	relatedness_score	entailment_judgment
	# df=df[~ (df.entailment_judgment == "NEUTRAL")]
	# print(df)
	df["entailment_judgment"] = df["entailment_judgment"].apply(lambda row:1 if row=="ENTAILMENT"  else 0)

	df[["sentence_A","sentence_B","entailment_judgment"]].to_csv(new,index=False,header=None,encoding="utf-8",sep="\t")