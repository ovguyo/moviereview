# Sentiment Analysis on IMDB Movie Reviews using Keras

The aim in this project is to classify IMDB movie reviews as "positive" or "negative". This is a binary classification task. I used Keras deep learning library to create an LSTM and CNN model to solve the task.

# 1. Sentiment Analysis with LSTM
## Dataset
A large Movie Review Dataset v1.0, which contains a set of 25,000 highly polar movie reviews for training and 25,000 for testing, has been used in this task. The dataset can be found the following link.
http://ai.stanford.edu/~amaas/data/sentiment/

The dataset already exists among keras datasets. It was imported with the following line and made ready to load.
```
from keras.datasets import imdb
```
## Getting Started
Functions and libraries required to create the model were imported.
```
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
```
## Loading IMDB Dataset
I kept the top 5000 words from the dataset, also splited it into two equal amount of training and test sets.

```
numpy.random.seed(7)
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

## Padding to the Same Length
The inputs should be in the same length. I specified maximum length as 500.
```
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_length)
```
## Word Embedding
Since movie reviews are actually sequences of words, there is need to encode them. Word embedding has been used to represent features of the words with semantic vectors and map each movie review into a real vector domain. I specified embedding vector length as 32.

```
embedding_vector_length=32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_length))
```
So the first layer of the model is embedding layer. This will use 32 length vector to represent each word.

## Creating LSTM Model
LSTM, which is an often used natural language processing technique for both sentiment analysis, text classification and machine translation, has been preferred to solve this task. LSTM is a special kind of recurrent neural network which is capable of learning long term dependencies [1]. LSTM is able to remember information for long periods of time as a default behavior.

```
model.add(LSTM(100, activation = 'tanh', recurrent_activation='hard_sigmoid', dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=512)
```
 
LSTM layer has 100 memory units. Activation function is tanh. I also used dropout to prevent overfitting in this layer. And since this is a binary classification, I needed to use a Dense layer containing only one neuron. Activation function in dense layer is sigmoid.
Adam, which is an adaptive learning method, was used as optimizer. Batch size was specified as 512.

## Evaluation with Test Set
The LSTM model was evaluated with test sets and accuracy was printed.

```
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```
Training and testing of the model for 5 epochs took ~2 hours. I obtained 87.74% accuracy.
