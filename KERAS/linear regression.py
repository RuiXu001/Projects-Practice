# imports

import keras
from keras.models import Sequential 
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100)
noise = np.random.normal(0,1,X.shape)
y = X*4+0.2+noise

plt.scatter(X,y)

model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(optimizer='sgd', loss='mse')

for step in range(5000):
    cost = model.train_on_batch(X,y)
    if step %100 == 0:
        print(step, cost)
        
w,b = model.layers[0].get_weights()
print(w, b)

y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred,'r-')
