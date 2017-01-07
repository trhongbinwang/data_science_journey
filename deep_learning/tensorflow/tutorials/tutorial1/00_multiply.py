#!/usr/bin/env python

import tensorflow as tf


def inputs_placeholder():
    a = tf.placeholder("float") # Create a symbolic variable 'a'
    b = tf.placeholder("float") # Create a symbolic variable 'b'
    return [a, b]

def model(a, b):
    y = tf.mul(a, b) # multiply the symbolic variables
    return y

def train(sess, y, a, b): 
        print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
        print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))


if __name__ == '__main__':
    ''' '''
    # define inputs
    [a, b] = inputs_placeholder()
    # define model
    y = model(a, b)
    with tf.Session() as sess: # create a session to evaluate the symbolic expressions
        # train model
        train(sess, y, a, b)