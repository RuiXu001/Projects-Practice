import keras
from keras.models import Sequential 
from keras.layers import Dense,Activation
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-3,3,300)
noise = np.random.normal(0,0.3,X.shape)
y = (X)**2+0.2+noise

plt.scatter(X,y)
plt.show()

model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='relu'))
model.add(Dense(units=1, activation='relu'))
sgd = SGD(lr=0.1)
model.compile(optimizer=sgd, loss='mse')

for step in range(3000):
    cost = model.train_on_batch(X,y)
    if step %100 == 0:
        print(step, cost)
        
w,b = model.layers[0].get_weights()
print(w, b)

y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred,'r-')
