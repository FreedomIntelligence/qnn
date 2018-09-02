import pandas as pd
for filename in ["train.txt" ]:
    df = pd.read_csv(filename,encoding="utf-8",sep="\t",names=["a","b","c","d","e"],quoting=3)
    print(df)
    df[["c","d","e"]].to_csv(filename,index=False,sep="\t",encoding="utf-8",header=None)