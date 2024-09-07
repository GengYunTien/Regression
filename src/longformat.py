import os
import pandas as pd

root = 'root'
R = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
V = ['a', 'b', 'c', 'd']
rv = [f'{r}{v}' for r in R for v in V]

dfAll = pd.DataFrame()
for i in rv:
    df = pd.read_csv(os.path.join(root, '2024-pre-train', i[:-1], i[-1:] + '.csv'))
    df = df.assign(odjID = i[:-1], condID = i[-1:])
    dfAll = pd.concat([dfAll, df], axis = 0, ignore_index = True)
dfAll.to_csv(os.path.join(root, '2024-training.csv'), index = False)
