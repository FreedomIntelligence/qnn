# -*- coding: utf-8 -*-

import pandas as pd
import codecs
filename = "log/acc20180812/20180812162841_546.log"

def process(string):
    result = dict()
    end =string.index("]]") +2
    start= string.index("[[") 

    record = string[:start] +"0"+ string[end:]
    
    print(record)
    record = record.replace("->","").replace(":","performance")
    tokens = [token for token in record.split(" ")[6:] if len(token.strip()) >0]
    print(tokens)
    for k,v in zip(tokens[::2],tokens[1::2]):
        result[k]=v
    return result
sample= ""
lines,records =[],[]
with codecs.open(filename,encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        
        
        if "->" in line and "INFO: running" not in line and "INFO: Comput" not in line and "INFO: Found " not in line and not ("[" in line and "]" in line and "[[" not in line  and "]]" not in line) :            

            sample = sample + line
            if "[[" not in line:
                record = process(sample)
                records.append(record)
                lines.append(sample)
                sample= ""
            

df=pd.DataFrame(records)
df.to_csv("rerults.csv",sep="\t",index=None,encoding="utf-8")
with codecs.open("demo","w",encoding="utf-8") as f:
    f.write("\n".join(lines))

print(df.groupby("dataset_name").apply(lambda group: group["performance"].max()))
print(df[["dataset_name","measurement_size","performance"]])

records = dict()
for i, row in df[["dataset_name","measurement_size","performance"]].set_index(keys=["dataset_name","measurement_size"]).iterrows():
    dataset = row.name[0]
    size = row.name[1]
    records.setdefault(dataset,[]);
    records[dataset].append((row.values[0],size))
print(records)
for i in range(len(records["CR"])):
    nums=[]
    for dataset in  ["CR","MPQA","SUBJ","MR","SST_2","SST_5","TREC"]:
        nums.append(records[dataset][-1*(i+1)][0])
    print(records[dataset][-1*(i+1)][1] + " &"+ " & ".join(nums) +"\\\\")
        
    