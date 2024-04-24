# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Part 1 - Identifying frauds with Self-Organizing Map

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = X.shape[1], sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration = 100)

# Visualizing the Results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,6)], mappings[(3,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# Part 2 - Going from unsupervised DL to supervised DL

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Creating ANN

# Importing libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
ann = Sequential()

# Adding the input layer and first hidden layer
ann.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = customers.shape[1]))

# Adding Output layer
ann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN into training set
ann.fit(customers, is_fraud, batch_size = 1, epochs = 5)

# Predicting and sorting the probability of fraud
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]










