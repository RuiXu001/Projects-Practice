#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import random
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras import optimizers
import keras
from keras import layers

#%%

tf.config.experimental.list_physical_devices()
# In[2]:


reviews = pd.read_csv('d:/msda/data245/project/tripadvisor_hotel_reviews.csv')
reviews


# In[3]:


df = reviews[reviews['Rating'] == 5]
df


# In[4]:


df = df.replace({r'\+': ''}, regex=True)


# In[5]:


#train on shorter reviews.  Already lots of data, easier to train on shorter ones too
#mask = (df['Review'].str.len() < 1000) 
#df2 = df.loc[mask]
df2 = df
len(df2)


# In[6]:


df2


# In[7]:


#shuffle the order of the reviews so we don't train on 100 Subway ones in a row
short_reviews=df2.sample(frac=1).reset_index(drop=True)


# In[8]:


#text = open('d:/msda/data245/project/fakereview/short_reviews_shuffle.txt').read()
text = ' '.join(short_reviews.Review)
print('Corpus length:', len(text))


# In[9]:


#only run this the first time, it will save a txt file on your computer
filename='d:/msda/data245/project/fakereview/short_reviews_shuffle.txt'
#short_reviews.to_csv(filename, header=None, index=None, sep=' ')
with open(filename, 'w', encoding="utf8") as f:
    f.write(text)
    f.flush()


# In[10]:


# List of unique characters in the corpus

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)

maxlen=60
step=1

char_indices
# In[11]:
chars
len(chars)

# In[12]:


def getDataFromChunk(txtChunk, maxlen=60, step=1):
    sentences = []
    next_chars = []
    for i in range(0, len(txtChunk) - maxlen, step):
        sentences.append(txtChunk[i : i + maxlen]) # each sentence contain 60 characters
        next_chars.append(txtChunk[i + maxlen])
    print('nb sequences:', len(sentences))
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        #print(i, sentence)
        for t, char in enumerate(sentence):
            #print(t, char)
            X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
    return [X, y]


# In[13]:


chunk = '111Always love this place and the prices are very reasonable I am never disappointed. 34This get Data From Chunk is necessary to process large data sets like the one we have.'
chunk = chunk.lower()
x,y = getDataFromChunk(chunk)
len(chunk)


# In[14]:


x.shape


# In[15]:


y.shape


# In[16]:


model = keras.models.Sequential()
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars)),return_sequences=True))
model.add(Dropout(0.2))
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(layers.Dense(len(chars), activation='softmax'))
#model.load_weights("D:\MSDA\data245\project\LSTM\lstm.hdf5")
print(model.summary())


# In[17]:


optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[18]:


# this saves the weights everytime they improve so you can let it train.  Also learning rate decay
filepath="lstm.hdf5" #-{epoch:02d}-{loss:.4f}
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
              patience=1, min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]


# In[19]:


def sample(preds, temperature=1.0):
    '''
    Generate some randomness with the given preds
    which is a list of numbers, if the temperature
    is very small, it will always pick the index
    with highest pred value
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[ ]:


#### This trains the model batching from the text file
#every epoch it prints out 300 characters at different "temperatures"
#temperature controls how random the characters sample: more temperature== more crazy (but often better) text
for iteration in range(1, 20): # 20
    print()
    print('-' * 50)
    print('Iteration', iteration)
    with open("d:/msda/data245/project/fakereview/short_reviews_shuffle.txt", 'r', encoding="utf8") as f:
        for chunk in iter(lambda: f.read(90000), ""):
            #print(chunk)
            X, y = getDataFromChunk(chunk)
            model.fit(X, y, batch_size=128, epochs=1, callbacks=callbacks_list)
            #model.fit(X, y, batch_size=128, epochs=1, callbacks=callbacks_list)
    
     # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.5, 0.8, 1.0]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 300 characters
        for i in range(300):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# In[48]:


#USE THIS TO TEST YOUR OUTPUT WHEN NOT/DONE TRAINING

# Select a text seed at random
'''start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index: start_index + maxlen]
print('--- Generating with seed: "' + generated_text + '"')

for temperature in [0.5, 0.8, 1.0]:
    print('------ temperature:', temperature)
    sys.stdout.write(generated_text)

        # We generate 300 characters
    for i in range(300):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]

        generated_text += next_char
        generated_text = generated_text[1:]

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()'''

#%%
model = keras.models.Sequential()
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars)),return_sequences=True))
model.add(Dropout(0.2))
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(layers.Dense(len(chars), activation='softmax'))
model.load_weights("D:\MSDA\data245\project\LSTM\lstm.hdf5")
print(model.summary())

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

X, y = getDataFromChunk(chunk)
history = model.fit(X, y, batch_size=128, epochs=1, callbacks=callbacks_list)

# this saves the weights everytime they improve so you can let it train.  Also learning rate decay
filepath="lstm.hdf5" #-{epoch:02d}-{loss:.4f}
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
              patience=1, min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]