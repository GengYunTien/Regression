import os
import pickle
import pandas as pd
import numpy as np

# load data
root = 'root'
trainingData = pd.read_csv(os.path.join(root, '2024-training.csv'))

# Create sequences
R = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
V = ['a', 'b', 'c', 'd']
exp = ['y01', 'y02', 'y03', 'y04', 'y05', 'y06', 'y07', 'y08', 'y09', 'y10']
rve = [[f'{r}', f'{v}', f'{e}'] for r in R for v in V for e in exp]
seq_x, seq_y = {}, {}
for i in rve:
    df = trainingData[(trainingData['odjID'] == int(i[0])) & (trainingData['condID'] == i[1])]
    df = df.loc[:, i[2]].values
    src = df[:50]
    trg = df[50:]
    seq_x[(f'{i[0]}{i[1]}', i[2])] = src
    seq_y[(f'{i[0]}{i[1]}', i[2])] = trg
fx = open("/root/seq_x.pkl", "wb")
fy = open("/root/seq_y.pkl", "wb")
pickle.dump(seq_x, fx)
pickle.dump(seq_y, fy)
