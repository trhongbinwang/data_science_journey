#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data


def load_data():
    ''' '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return [trX, trY, teX, teY]    

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def inputs_placeholder(): 
    
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    return [X, Y, p_keep_input, p_keep_hidden]

def model(X, Y, p_keep_input, p_keep_hidden):
    ''' '''
    # share variables -- can be auto infer
    w_h = init_weights([784, 625])
    w_h2 = init_weights([625, 625])
    w_o = init_weights([625, 10])
    # basic graph
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    py_x = tf.matmul(h2, w_o)
    # outputs
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)
    return [train_op, predict_op]
        
def train(sess, trX, trY, teX, teY, X, Y, p_keep_input, p_keep_hidden, train_op, predict_op):
    
    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))

if __name__ == '__main__':
    ''' '''
    # load data
    [trX, trY, teX, teY] = load_data()
    # define inputs
    [X, Y, p_keep_input, p_keep_hidden] = inputs_placeholder()
    # define model
    [train_op, predict_op] = model(X, Y, p_keep_input, p_keep_hidden)
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
    
        # train
        train(sess, trX, trY, teX, teY, X, Y, p_keep_input, p_keep_hidden, train_op, predict_op)
        
    
















