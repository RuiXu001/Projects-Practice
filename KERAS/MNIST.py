# imports

import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.regularizers import l2


import numpy as np
import matplotlib.pyplot as plt

# load data
(X_train, y_train),(X_test, y_test) = mnist.load_data()
print('X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

X_train = X_train.reshape(X_train.shape[0], -1)/255
X_test = X_test.reshape(X_test.shape[0], -1)/255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# build model
model = Sequential()
model.add(Dense(units=100, input_dim=X_train.shape[1], bias_initializer='one', activation='relu', kernel_regularizer=l2(0.0003)))
model.add(Dropout(0.2))
model.add(Dense(units=50, bias_initializer='one', activation='relu', kernel_regularizer=l2(0.0003)))
model.add(Dropout(0.2))
model.add(Dense(units=10, bias_initializer='one', activation='softmax', kernel_regularizer=l2(0.0003)))

#sgd=SGD(lr=0.3)
adam=Adam(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# train
history = model.fit(X_train, y_train, batch_size=64, epochs=30)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# evaluate
loss, acc = model.evaluate(X_test, y_test)
print(loss, acc)
