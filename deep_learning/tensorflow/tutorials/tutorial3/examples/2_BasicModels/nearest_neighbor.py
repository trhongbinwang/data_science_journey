'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    # Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    # In this example, we limit mnist data
    Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
    Xte, Yte = mnist.test.next_batch(200) #200 for testing
    return [Xtr, Ytr, Xte, Yte]

def inputs_placeholder():
        
    # tf Graph Input
    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])
    return [xtr, xte]

def model(xtr, xte):
    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), axis=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)
    return pred


def train(sess, Xtr, Ytr, Xte, Yte, xtr, xte, pred):
    '''
    data: Xtr, Ytr, Xte, Yte
    graph: xtr, xte, pred
    
    '''
    accuracy = 0.
    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    ''' '''
    # load data
    [Xtr, Ytr, Xte, Yte] = load_data()
    # define inputs
    [xtr, xte] = inputs_placeholder()
    # model
    pred = model(xtr, xte)
    # Launch the graph
    with tf.Session() as sess:
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # train
        train(sess, Xtr, Ytr, Xte, Yte, xtr, xte, pred)
    
    
    
    




