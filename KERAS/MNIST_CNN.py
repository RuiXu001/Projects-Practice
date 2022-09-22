# imports

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.regularizers import l2


import numpy as np
import matplotlib.pyplot as plt

# load data
(X_train, y_train),(X_test, y_test) = mnist.load_data()
print('X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

X_train = X_train.reshape(-1,28,28,1)/255
X_test = X_test.reshape(-1,28,28,1)/255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train: {}\ny_train: {}\nX_test: {}\ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

#model
model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2,'same'))
model.add(Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2,'same'))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

#sgd=SGD(lr=0.3)
adam=Adam(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#train
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
