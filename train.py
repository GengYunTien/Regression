from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, random, math, pickle

# Set Seed for reproducibility
seed = 12040904
np.random.seed(seed)

# load training data
with open('/root/seq_x.pkl', 'rb') as fx:
    seq_x = pickle.load(fx)
with open('/root/seq_y.pkl', 'rb') as fy:
    seq_y = pickle.load(fy)

R = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
V = ['a', 'b', 'c', 'd']
exp = ['y01', 'y02', 'y03', 'y04', 'y05', 'y06', 'y07', 'y08', 'y09', 'y10']
train_rve = [(f'{r}{v}', f'{e}') for r in R for v in V for e in exp]
train_x = np.array([seq_x[key] for key in train_rve])
train_y = np.array([seq_y[key] for key in train_rve])

# load testing data
root = 'root'
df = pd.read_csv(os.path.join(root, '2024-testing.csv'))
df = df.loc[:, 'y01':'y10'].values
df = df[:50, :]
df_test = df.transpose()

# standardization
scaler_x = StandardScaler()
scaler_y = StandardScaler()

train_x = scaler_x.fit_transform(train_x)
train_y = scaler_y.fit_transform(train_y)

test_x = scaler_x.transform(df_test)

# model
model = MLPRegressor(
    hidden_layer_sizes = (256, 1024),
    activation = 'relu',
    solver = 'adam',
    max_iter = 800,
    random_state = seed
)

# training
model.fit(train_x, train_y)

# testing
y_pred = model.predict(test_x)

# inverse standardization
y_pred = scaler_y.inverse_transform(y_pred)

# plot function definition
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

y_pred = y_pred.transpose()

plt.figure(figsize = (20, 10))
x = np.arange(50)
plt.plot(x, df[:, 0], color = 'steelblue', alpha = 0.6, label = 'True values')
for k in range(1, df.shape[1]):
    plt.plot(x, df[:, k], color = 'steelblue', alpha = 0.6)

x = np.arange(50, 4000)
plt.plot(x, y_pred[:, 0], color = 'sandybrown', alpha = 0.6, label = 'Prediction values')
for k in range(1, y_pred.shape[1]):
    plt.plot(x, y_pred[:, k], color = 'sandybrown', alpha = 0.6)

plt.title('testing')
plt.legend()
plt.savefig('results', bbox_inches = 'tight')

# answer.csv
testingData = pd.DataFrame(y_pred, columns = [f'y0{i+1}' for i in range(10)])
testingData.insert(0, 'id', np.arange(51, 4001))
testingData.to_csv(os.path.join(root, 'answer.csv'), index = False)
