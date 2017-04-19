'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function

from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.layers import Input, Embedding
from tensorflow.contrib.keras.python.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.contrib.keras.python.keras.datasets import imdb

# set hype parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2


def load_data():
    '''
    '''
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


def cnn_model_fn():
    ''' '''
    print('Build model...')
    inputs = Input(shape=(maxlen,), dtype='int32') # a index sequence with lenght = maxlen
    x = Embedding(  max_features,
                    embedding_dims,
                    input_length=maxlen)(inputs)
    x = Dropout(0.2)(x)
    x = Conv1D(filters,
                kernel_size,
                padding='valid',
                activation='relu',
                strides=1)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(hidden_dims)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train(data, model):
    ''' '''
    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

if __name__ == '__main__':
    ''' '''
    data = load_data()
    model = cnn_model_fn()
    train(data, model)
