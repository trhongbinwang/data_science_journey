'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.layers import Embedding
from tensorflow.contrib.keras.python.keras.layers import LSTM
from tensorflow.contrib.keras.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.contrib.keras.python.keras.datasets import imdb

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

def load_data():
    ''' '''
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return [x_train, y_train, x_test, y_test]


def cnn_lstm_model():
    ''' '''
    print('Build model...')
    inputs = Input(shape=(maxlen, ), dtype='int32')
    x = Embedding(max_features, embedding_size, input_length=maxlen)(inputs)
    x = Dropout(0.25)(x)
    x = Conv1D(filters,
                kernel_size,
                padding='valid',
                activation='relu',
                strides=1)(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = LSTM(lstm_output_size)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train(data, model):
    ''' '''
    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__== '__main__':
    ''' '''
    data = load_data()
    model = cnn_lstm_model()
    train(data, model)
