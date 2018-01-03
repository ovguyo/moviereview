# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:05:19 2017

@author: Övgü
"""

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, Flatten, Dropout
from keras.layers import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)

# load the dataset but only keep the top 5000 words
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

#Padding the sequences to the same length
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_length)

#Word Embedding
embedding_vector_length=32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_length))

#LSTM Model
model.add(Conv1D(64, 3, border_mode= 'same'))
model.add(Conv1D(32, 3, border_mode= 'same'))
model.add(Conv1D(16, 3, border_mode= 'same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=256)

#Evaluation on the Test Set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))