# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:59:36 2021

@author: xurui
"""

#%%
import os
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from keras.models import load_model

#%%
basedir = r'd:/msda/data255/parking'
patchespath = r'd:/msda/data255/parking/patches/'
os.chdir(basedir)

df = pd.read_csv('CNRPark+EXT.csv')
df.columns
df
X = list(patchespath+df.image_url)
Y = list(df.occupancy)
#Y = keras.utils.to_categorical(Y, num_classes=2)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.002,random_state=42)

print('Train ', np.shape(x_train), np.shape(y_train))
print('Validation ', np.shape(x_valid), np.shape(y_valid))
print('Test ', np.shape(x_test), np.shape(y_test))

x_train[0]

def getdata(x, y, start_n, end_n):
    xtrain = []
    for i in range(start_n, end_n):
        #path = patches+x[i]
        # print(path)
        img = cv2.imread(x[i], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150,150)) # some images are 150*126, so all the images should be resized
        #print(np.shape(img))
        xtrain.append(img/255)
    ylabel = y[start_n: end_n]
    #xtrain  = np.array(xtrain)
    #ylabel = np.array(ylabel)
    return np.array(xtrain), np.array(ylabel)

xtrain, ylabel = getdata(x_train, y_train, 0, 10000)
xvalid, yvalid = getdata(x_valid, y_valid, 0, len(x_valid))
xtest, ytest = getdata(x_test, y_test, 0, len(x_test))

epoches = 10
batch_size = 32

def show(x,y, index_img):
    plt.imshow(x[index_img])
    plt.title(y[index_img])
show(xtrain, ylabel, 1)
show(xtrain, ylabel, 2)
show(xtrain, ylabel, 3)
show(xtrain, ylabel, 4)
show(xtrain, ylabel, 5)


#%% AlexNet
# alexnet
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4),  activation='relu', 
                        input_shape=(150,150,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(xtrain,ylabel,
          epochs=epoches,
          batch_size = batch_size,
          validation_data=(xvalid, yvalid),
          validation_freq=1)
#%% train the alexnet using all the training data which contain 120K records
# loss: 0.0884 - accuracy: 0.9810 - val_loss: 0.1047 - val_accuracy: 0.9684
for i in range(12):
    xtrain, ylabel = getdata(x_train, y_train, i*10000, (i+1)*10000)
    model.fit(xtrain,ylabel, epochs=epoches,
              batch_size = batch_size, validation_data=(xvalid, yvalid))
xtrain, ylabel = getdata(x_train, y_train, 120000, 125786)
model.fit(xtrain,ylabel,
          epochs=epoches,
          batch_size = batch_size,
          validation_data=(xvalid, yvalid),
          validation_freq=1)
#%% Sunny - alexnet 
# Rain Train  (60487,) (60487,) Validation  (122,) (122,) Test  (15153,) (15153,)
df = pd.read_csv('CNRPark+EXT.csv')
df.columns
df = df[df['weather'] == 'S']
df
X = list(patchespath+df.image_url)
Y = list(df.occupancy)
#Y = keras.utils.to_categorical(Y, num_classes=2)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.002,random_state=42)

print('Train ', np.shape(x_train), np.shape(y_train))
print('Validation ', np.shape(x_valid), np.shape(y_valid))
print('Test ', np.shape(x_test), np.shape(y_test))

xtrain, ylabel = getdata(x_train, y_train, 0, 20000)
xtrain, ylabel = getdata(x_train, y_train, 20000, 40000)
xtrain, ylabel = getdata(x_train, y_train, 40000, len(x_train))


xvalid, yvalid = getdata(x_valid, y_valid, 0, len(x_valid))
xtest, ytest = getdata(x_test, y_test, 0, len(x_test))

np.shape(xtrain)
# after 10 epochs, the model reach  loss: 0.0840 - accuracy: 0.9733 - val_loss: 0.0960 - val_accuracy: 0.9577
model.save('AlexNet_S.hdf5')
#%% Rain - alexnet 
# Rain Train  (29974,) (29974,) Validation  (61,) (61,) Test  (7509,) (7509,)
df = pd.read_csv('CNRPark+EXT.csv')
df.columns
df = df[df['weather'] == 'R']
df
X = list(patchespath+df.image_url)
Y = list(df.occupancy)
#Y = keras.utils.to_categorical(Y, num_classes=2)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.002,random_state=42)

print('Train ', np.shape(x_train), np.shape(y_train))
print('Validation ', np.shape(x_valid), np.shape(y_valid))
print('Test ', np.shape(x_test), np.shape(y_test))

xtrain, ylabel = getdata(x_train, y_train, 0, 10000)
xtrain, ylabel = getdata(x_train, y_train, 10000, len(x_train))

xvalid, yvalid = getdata(x_valid, y_valid, 0, len(x_valid))
xtest, ytest = getdata(x_test, y_test, 0, len(x_test))

np.shape(xtrain)
# after 10 epochs, the model reach  loss: 0.1669 - accuracy: 0.9640 - val_loss: 0.0533 - val_accuracy: 0.9672
model.save('AlexNet_R.hdf5')
#%% Overcast - alexnet 
# Rain Train  (35323,) (35323,) Validation  (71,) (71,) Test  (8849,) (8849,)
df = pd.read_csv('CNRPark+EXT.csv')
df.columns
df = df[df['weather'] == 'O']
df
X = list(patchespath+df.image_url)
Y = list(df.occupancy)
#Y = keras.utils.to_categorical(Y, num_classes=2)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.002,random_state=42)

print('Train ', np.shape(x_train), np.shape(y_train))
print('Validation ', np.shape(x_valid), np.shape(y_valid))
print('Test ', np.shape(x_test), np.shape(y_test))

xtrain, ylabel = getdata(x_train, y_train, 0, 10000)
xtrain, ylabel = getdata(x_train, y_train, 10000, len(x_train))

xvalid, yvalid = getdata(x_valid, y_valid, 0, len(x_valid))
xtest, ytest = getdata(x_test, y_test, 0, len(x_test))

np.shape(xtrain)
# after 10 epochs, the model reach  loss: 0.0335 - accuracy: 0.9902 - val_loss: 0.0470 - val_accuracy: 0.9859
model.save('AlexNet_O.hdf5')

#%%
model.save('AlexNet_all.hdf5')
#%%
df = pd.DataFrame(model.predict(xtest[:100]), columns=['Pred'])
df['Actual'] = ytest[:100]
df['Prediction'] = [1 if i >0.1 else 0 for i in df['Pred']]
df[df['Actual'] != df['Prediction']]
df

#%% Try one single layer cnn model
model_o = Sequential()
model_o.add(Conv2D(filters = 256, kernel_size = (5,5),padding = 'Same',
                   activation ='relu', input_shape = (150,150,3)))
model_o.add(keras.layers.BatchNormalization())
model_o.add(MaxPooling2D(pool_size=(2,2)))
model_o.add(Flatten())
model_o.add(Activation('relu'))
model_o.add(Dense(1, activation = "sigmoid"))

model_o.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
model_o.summary()
model_o.fit(xtrain,ylabel,
          epochs=epoches,
          batch_size = batch_size,
          validation_data=(xvalid, yvalid),
          validation_freq=1)



model_o.predict(xtrain[:10])
model_o.save('single.hdf5')

#%% set the cutoff
model = load_model(basedir + '\model\cnn_model.hdf5')
model.summary()
dfr = pd.DataFrame(model.predict(xtest), columns=['Pred'])
dfr['Prediction'] = [1 if i >0.1 else 0 for i in dfr['Pred']]
dfr['Actual'] = ytest
for i in dfr:
    print(i)
plt.plot([])

fpr, tpr, thresholds = roc_curve(np.array(ytest),np.array(dfr['Pred']))

dfr[dfr['Actual']==1]

from sklearn.metrics import roc_curve, auc
plt.scatter(dfr[dfr['Actual']==1]['Pred'], dfr[dfr['Actual']==1].index)
plt.plot(fpr, tpr, 'k')
plt.plot([0,1],[0,1], 'r--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()

dfr[dfr['Actual']!=dfr['Prediction']]

#%% VIDEO

video_path = 'd:\\msda\\data255\\parking\\video.mp4'
# Read the video from specified path
cam = cv2.VideoCapture(video_path)
  
try:
      
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 0
  
ret,frame = cam.read()
plt.imshow(frame)
plt.title(ret)
fps = 10
while(True):
      
    # reading from frame
    for i in range(fps):
        ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)
  
        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()


