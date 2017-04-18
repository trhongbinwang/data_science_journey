'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.objectives import categorical_crossentropy


# add prefix in import line. from tensorflow.contrib.keras.python.

# hyperparameter
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


def load_data():
    '''
    load data

    :return:
    '''
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return [x_train, y_train, x_test, y_test]


def inputs_placeholder():
    '''
    '''
    img = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.float32, shape=(None, 10))
    return [img, labels]


def cnn_model_tf(img, labels):
    '''
    defin the model in tf way

    '''
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    pred = Dense(num_classes, activation='softmax')(x)
    loss = tf.reduce_mean(categorical_crossentropy(labels, pred))


def cnn_model_fn():
    '''
    define the model in function way

    '''
    # input shape is (img_rows, img_cols, fea_channel)
    inputs = Input(shape=(img_rows, img_cols, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    pred = Dense(num_classes, activation='softmax')(x)
    # small change of Model parameters names, now is inputs, outputs
    model = Model(inputs=inputs, outputs=pred)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def train(data, model):
    '''
    train the model
    Args: data, model
    '''
    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main_tf():
    '''
    main func in tf way
    '''
    # initialiaze tf session
    sess = tf.Session()
    K.set_session(sess)  # register tf session with keras


def main_fn():
    '''
    main func in function way
    '''

    data = load_data()
    model = cnn_model_fn()
    train(data, model)

if __name__ == '__main__':
    '''

    '''
    main_fn()
