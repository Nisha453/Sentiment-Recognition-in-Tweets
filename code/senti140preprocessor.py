import csv

import pandas as pd
import numpy as np



df = pd.read_csv('../dataset/senti140/training.1600000.processed.noemoticon.csv')
df=df.sample(n=15000)
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]
test.to_csv("../dataset/senti140/subtest.csv",index=False)
train.to_csv("../dataset/senti140/subtrain.csv",index=False)
c=0

c=0
with open("../dataset/senti140/subtrain.csv") as f:
    with open("../dataset/senti140/subtrainfc.csv","w",newline="") as fw:
        cr=csv.reader(f)
        cw=csv.writer(fw)
        for line in cr:
            c+=1
            print(c)
            tweet_id=int(line[1])
            sentiment=0 if line[0]=="0" else 1
            tweet="\""+line[5]+"\""
            cw.writerow([tweet_id,sentiment,tweet])
with open("../dataset/senti140/subtest.csv") as f:
    with open("../dataset/senti140/subtestfc.csv","w",newline="") as fw:
        cr=csv.reader(f)
        cw=csv.writer(fw)
        for line in cr:
            c+=1
            print(c)
            tweet_id=int(line[1])
            sentiment=0 if line[0]=="0" else 1
            tweet="\""+line[5]+"\""
            cw.writerow([tweet_id,sentiment,tweet])