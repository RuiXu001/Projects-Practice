# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 00:07:08 2020

@author: xurui
"""

import sys
import numpy as np
import keras
from keras import layers
# this text file can be any text, as long as it contains text longer than 'maxlen' defined below
#text=open('../seed_text.txt').read()
text = 'the staff is nice and friendly. we stayed about 1 week. the room is clean and bed is comfortable'
#chars=['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
#chars = [' ', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '9', 'a', 'b', 'c',
# 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
# 'u', 'v', 'w', 'x', 'y', 'z']
chars = [' ',
 "'",
 '*',
 ',',
 '-',
 '.',
 '/',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 ':',
 '=',
 '\\',
 '_',
 '`',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z',
 '~',
 '\x81',
 '\x82',
 '\x8a',
 '\x8c',
 '\x8d',
 '\x8f',
 '\x90',
 '\x91',
 '\x93',
 '\x94',
 '\x96',
 '\x99',
 '\x9d',
 '\xa0',
 '¢',
 '£',
 '©',
 '«',
 '±',
 '´',
 'À',
 'Â',
 'Ä',
 'Ç',
 'È',
 'Ù',
 'Û',
 'Ü',
 'à',
 'â',
 'ã',
 'ä',
 'å',
 'æ',
 'ç',
 'è',
 'é',
 'ê',
 'ë',
 'î',
 'ö']
len(chars)
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)
maxlen=60
step=1

'''model = keras.models.Sequential()
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars)),return_sequences=True))
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))'''

model = keras.models.Sequential()
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars)),return_sequences=True))
model.add(Dropout(0.2))
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(layers.Dense(len(chars), activation='softmax'))
model.load_weights("D:\MSDA\data245\project\LSTM\lstm.hdf5")
optimizer = keras.optimizers.Adam(lr=0.0002)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True)
#from IPython.display import Image
#Image(filename='model.png') 
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

def random_reviews():
    # sample a start index
    start_index = np.random.randint(0, len(text) - maxlen - 1)
    # the initial sampled text with maxlen long
    generated_text = text[start_index: start_index + maxlen]
    print('Coming up with several reviews for you...')

    for temperature in [0.8]:
        sys.stdout.write(generated_text)

        # We generate 600 characters
        for i in range(600):
            sampled = np.zeros((1, maxlen, len(chars)))
            # Turn each char to char index.
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            # Predict next char probabilities
            preds = model.predict(sampled, verbose=0)[0]
            # Add some randomness by sampling given probabilities.
            next_index = sample(preds, temperature)
            # Turn char index to char.
            next_char = chars[next_index]
            # Append char to generated text string
            generated_text += next_char
            # Pop the first char in generated text string.
            generated_text = generated_text[1:]
            # Print the new generated char.
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print(generated_text)
random_reviews()