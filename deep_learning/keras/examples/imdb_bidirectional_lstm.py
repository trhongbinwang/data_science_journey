'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.contrib.keras.python.keras.datasets import imdb

# hyperparameters
max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32


def load_data():
    ''' '''
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print("Pad sequences (samples x time)")
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return [x_train, y_train, x_test, y_test]


def bidirectional_model():
    inputs = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(max_features, 128, input_length=maxlen)(inputs)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


def train(data, model):
    ''' '''
    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=4,
              validation_data=[x_test, y_test])

if __name__ == '__main__':
    ''' '''
    data = load_data()
    model = bidirectional_model()
    train(data, model)