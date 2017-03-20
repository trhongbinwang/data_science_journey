# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:48:09 2017

keras as tf api
always use as an tf api

In this case, we use Keras only as a syntactical shortcut to generate an op 
that maps some tensor(s) input to some tensor(s) output, and that's it.

@author: hongbin
"""

import tensorflow as tf

from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy # iy just a function

# initialiaze tf session
sess = tf.Session()
K.set_session(sess) # register tf session with keras

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

# ----- use Keras layers to speed up the model definition process!!!


# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation


loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# load data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# train
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_op.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

# evaluate

acc_value = accuracy(labels, preds)
with sess.as_default():
    print( acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}))





#Calling a Keras model on a TensorFlow tensor
#
#A Keras model acts the same as a layer, and thus can be called on TensorFlow tensors:
from keras.models import Sequential

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

# this works! 
x = tf.placeholder(tf.float32, shape=(None, 784))
y = model(x)












