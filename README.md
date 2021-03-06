# Sentiment Analysis on IMDB Movie Review Dataset using Keras

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
I kept the top 5000 words from the dataset, also splited the dataset into two equal parts which are training and test sets.

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

The LSTM model was evaluated on test sets and accuracy, precision, recall, F-score Confusion Matrix: were printed.

```
score, acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Test score: %.2f%%" % (score*100))
from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(np.array(X_test))
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
target_names = ['pos', 'neg']
cnf_matrix_test = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=target_names))
print(cnf_matrix_test)
df_cm = pd.DataFrame(cnf_matrix_test, range(2), range(2))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='d')

```
Training and testing of the model for 2 epochs took ~1 hour. The accuracy obtained is 86.41 %.

```
Epoch 1/2
25000/25000 [==============================] - 858s - loss: 0.5865 - acc: 0.6908     
Epoch 2/2
25000/25000 [==============================] - 1054s - loss: 0.3168 - acc: 0.8680     
Accuracy: 86.41%
Test score: 31.82%
```
```
               precision    recall  f1-score   support

        pos       0.83      0.92      0.87     12500
        neg       0.91      0.81      0.86     12500

avg / total       0.87      0.86      0.86     25000
```

Confusion matrix:

![lstmcm](https://user-images.githubusercontent.com/35049725/34943885-c766acc6-f9fd-11e7-9e7c-e57a713babb3.png)

According to the confusion matrix, 10112 positive reviews were correctly predicted (True Positive) and 11491 negative samples were correctly predicted (True Negative). And number of incorrect predictions are 2388 (False Positive) and 1009 (False Negative).

Misclassification rate is calculated and found as 0.13588.
```
TP = cnf_matrix_test[1, 1]
TN = cnf_matrix_test[0, 0]
FP = cnf_matrix_test[0, 1]
FN = cnf_matrix_test[1, 0]

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
```

# 2. Sentiment Analysis with CNN
CNN is a powerful deep learning algorithm for solving particulary image classification and many other tasks. Basicially, CNN trains filters as feature identifiers and does element-wise multiplications in convolutional layers to get a feature map representing the features.

## Getting Started
Functions and libraries for creating a CNN model were imported.
```
import numpy as np
import seaborn as sn
import pandas as pd
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D
from keras.layers import Embedding
from keras.preprocessing import sequence
```
## Loading the Dataset, Padding and Word Embedding

These are same with the operations before creating LSTM model.
```
np.random.seed(7)
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

Since inputs are sequences in this task, I used 1D convolutional layers in the model. The model consists of 4 convolutional layers containing 32 neurons. I specified kernel size as 3. Dense layer containing 64 neurons was also added. Batch size was specified as 32 and adam optimizer was used.
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

Evaluation on the test set, calculation of accuracy, precision, recall, F-score and visualization of Confusion Matrix:
```
score, acc = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (acc*100))
print("Test score: %.2f%%" % (score*100))
from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(np.array(X_test))
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
target_names = ['pos', 'neg']
cnf_matrix_test = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=target_names))
print(cnf_matrix_test)
df_cm = pd.DataFrame(cnf_matrix_test, range(2), range(2))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='d')
```

For 5 epochs, results are in below. The accuracy obtained is 84.67%.
```
Epoch 1/3
25000/25000 [==============================] - 101s - loss: 0.3829 - acc: 0.8176   
Epoch 2/3
25000/25000 [==============================] - 100s - loss: 0.2427 - acc: 0.9042   
Epoch 3/3
25000/25000 [==============================] - 107s - loss: 0.1590 - acc: 0.9428      
Accuracy: 84.67%
Test score: 39.92%
```
Precision, recall and F-score are printed.

```
             precision    recall  f1-score   support

        pos       0.85      0.85      0.85     12500
        neg       0.85      0.85      0.85     12500

avg / total       0.85      0.85      0.85     25000
```
Confusion matrix can be seen below.

![cnnsa1](https://user-images.githubusercontent.com/35049725/34922190-d7f5b5f8-f98c-11e7-8d24-1e022963d035.png)

According to the confusion matrix, 10566 positive reviews were correctly predicted (True Positive) and 10601 negative samples were correctly predicted (True Negative). And number of incorrect predictions are 1899 (False Positive) and 1934 (False Negative).

To calculate misclassification rate:
```
TP = cnf_matrix_test[1, 1]
TN = cnf_matrix_test[0, 0]
FP = cnf_matrix_test[0, 1]
FN = cnf_matrix_test[1, 0]
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)
```
It was calculated as 0.15332.

Since there is a big difference between training and test accuracy, I changed the model a little by adding pooling layers, dropout and reducing the number of neurons in the convolutional layers so as to boost the model and prevent overfitting. The new model can be seen below.
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

After the evaluation of this model, I observed that the accuracy was a little increased and reached to 88.01 % value. Training and evaluation took almost 10 minutes.
```
Epoch 1/3
25000/25000 [==============================] - 53s - loss: 0.4276 - acc: 0.7770      
Epoch 2/3
25000/25000 [==============================] - 51s - loss: 0.2580 - acc: 0.8989     
Epoch 3/3
25000/25000 [==============================] - 50s - loss: 0.2241 - acc: 0.9151     
Accuracy: 88.01%
Test score: 28.89%   
```
```
              precision    recall  f1-score   support

        pos       0.88      0.88      0.88     12500
        neg       0.88      0.88      0.88     12500

avg / total       0.88      0.88      0.88     25000

```
Confusion Matrix:

![cnnsa2](https://user-images.githubusercontent.com/35049725/34922329-b6726730-f98e-11e7-95dc-128af4ac8a48.png)

According to the confusion matrix, 10963 positive reviews were correctly predicted (True Positive) and 11040 negative samples were correctly predicted (True Negative). And number of incorrect predictions are 1460 (False Positive) and 1537 (False Negative).
Misclassification rate was calculated as. 0.11988.

# Conclusion

The table comparing the architectures and results is given below.

![tablo](https://user-images.githubusercontent.com/35049725/34944345-84d0ee60-f9ff-11e7-8ab1-0f6fc0f678f5.png)

 According to the table, the best performance based on accuracy and training duration belongs to the second CNN architecture.
 
 # DEMO
 
 Weights and pretrained CNN model were saved to model_cnn.h5 and model_cnn.json files. 
 You can test the reviews that you will give as an input to the text box using demo_cnn(2).ipynb. The performance is not so good when you write short sentences. 
Here is an example to copy and test some reviews for Zodiac movie.
https://mubi.com/films/zodiac

## References

[1] http://ai.stanford.edu/~amaas/data/sentiment/

[2] http://colah.github.io/posts/2015-08-Understanding-LSTMs/



