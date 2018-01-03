# Sentiment Analysis on IMDB Movie Reviews using Keras

The aim in this project is to classify IMDB movie reviews as "positive" or "negative". This is a binary classification task. I used Keras deep learning library to create an LSTM and CNN model to solve the task.

# 1. Sentiment Analysis with LSTM
## Dataset
A large Movie Review Dataset v1.0, which contains a set of 25,000 highly polar movie reviews for training and 25,000 for testing, has been used in this task. The dataset can be found the following link [1].
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
LSTM, which is an often used natural language processing technique for both sentiment analysis, text classification and machine translation, has been preferred to solve this task. LSTM is a special kind of recurrent neural network which is capable of learning long term dependencies [2]. LSTM is able to remember information for long periods of time as a default behavior.

```
model.add(LSTM(100, activation = 'tanh', recurrent_activation='hard_sigmoid', dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=256)
```
 
LSTM layer has 100 memory units. Activation function is tanh. I also used dropout to prevent overfitting in this layer. And since this is a binary classification, I needed to use a Dense layer containing only one neuron. Activation function in dense layer is sigmoid.
Adam, which is an adaptive learning method, was used as optimizer. Batch size was specified as 256.

## Evaluation on the Test Set
The LSTM model was evaluated with test sets and accuracy was printed.

```
score, acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Test score: %.2f%%" % (score*100))
```
Training and testing of the model for 2 epochs took ~1 hour. The accuracy obtained is 88.08 %.

```
Epoch 1/2
25000/25000 [==============================] - 819s - loss: 0.5369 - acc: 0.7096     
Epoch 2/2
25000/25000 [==============================] - 1051s - loss: 0.2865 - acc: 0.8846    
Accuracy: 88.08%
```

# 2. Sentiment Analysis with CNN

## Getting Started
Functions and libraries for creating a CNN model were imported.
```
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D
from keras.layers import Embedding
from keras.preprocessing import sequence
```
## Loading the Dataset, Padding and Word Embedding

These are same with the operations before creating LSTM model.
```
numpy.random.seed(7)
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_length)
embedding_vector_length=32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_length))
```
## Creating CNN Model and Evaluation

CNN is a very powerful algorithm for solving particulary image classification tasks. Since inputs are sequences in this task, I used 1D convolutional layers in the model. The model consists of 4 convolutional layers containing 32 neurons. I specified kernel size as 3. Dense layer containing 64 neurons was also added. Batch size was specified as 32 and adam optimizer was used.
```
model.add(Conv1D(32, kernel_size= 3, padding= 'same', input_shape=(max_length, embedding_vector_length)))
model.add(Conv1D(32, kernel_size= 3, padding= 'same'))
model.add(Conv1D(32, kernel_size= 3, padding= 'same'))
model.add(Conv1D(32, kernel_size= 3, padding= 'same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
```

Evaluation on the test set:
```
score, acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (acc*100))
print("Test score: %.2f%%" % (score*100))
```

For 5 epochs, results are in below. The accuracy obtained is 85.25%.
```
Epoch 1/5
25000/25000 [==============================] - 107s - loss: 0.3828 - acc: 0.8177     
Epoch 2/5
25000/25000 [==============================] - 104s - loss: 0.2427 - acc: 0.9041     
Epoch 3/5
25000/25000 [==============================] - 103s - loss: 0.1591 - acc: 0.9433     
Epoch 4/5
25000/25000 [==============================] - 101s - loss: 0.0877 - acc: 0.9720     
Epoch 5/5
25000/25000 [==============================] - 102s - loss: 0.0606 - acc: 0.9818     
Accuracy: 85.25%
Test score: 53.78%
```

Since there is a big difference between training and test accuracy, I changed the model a little different by adding pooling layers, dropout and reducing the number of neurons in the convolutional layers and number of epochs so as to boost the model and prevent overfitting. The new model can be seen below.
```
model.add(Conv1D(32, kernel_size= 3, padding= 'same', input_shape=(max_length, embedding_vector_length)))
model.add(Conv1D(32, kernel_size= 3, padding= 'same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16, kernel_size= 3, padding= 'same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(16, kernel_size= 3, padding= 'same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32)
```

After the evaluation of this model, I observed that the accuracy was a little increased and reached 87.18% value. Training and evaluation took almost 8 minutes.
```
Epoch 1/3
25000/25000 [==============================] - 50s - loss: 0.4830 - acc: 0.7306      
Epoch 2/3
25000/25000 [==============================] - 49s - loss: 0.2615 - acc: 0.9010     
Epoch 3/3
25000/25000 [==============================] - 49s - loss: 0.2245 - acc: 0.9168     
Accuracy: 87.18%
Test score: 30.28%
```

In conclusion, it is obvious that CNN has an advantage of speed in comparison with LSTM. Despite of that, the accuracy obtained with LSTM (88.08 %) is a little better than accuracy of the CNN model (87.18 %).

## References

[1] http://ai.stanford.edu/~amaas/data/sentiment/
[2] http://colah.github.io/posts/2015-08-Understanding-LSTMs/



