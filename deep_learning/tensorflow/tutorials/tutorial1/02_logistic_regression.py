#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def load_data():
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return [trX, trY, teX, teY]

def inputs_placeholder():
    
    X = tf.placeholder("float", [None, 784]) # create symbolic variables
    Y = tf.placeholder("float", [None, 10])
    return [X, Y]

def model(X, Y):
    w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression
    
    py_x = tf.matmul(X, w)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
    predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression
    return [train_op, predict_op]
    
def train(sess, trX, trY, teX, teY, X, Y, train_op, predict_op):
    ''' train the model '''
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))



if __name__ == '__main__':
    ''' '''
    # load data
    [trX, trY, teX, teY] = load_data()
    # define inputs
    [X, Y] = inputs_placeholder()
    # define model
    [train_op, predict_op] = model(X, Y)
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        # train
        train(sess, trX, trY, teX, teY, X, Y, train_op, predict_op)
    
    
    


