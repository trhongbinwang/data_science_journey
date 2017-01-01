#!/usr/bin/env python

import tensorflow as tf
import numpy as np

def create_data():
    
    trX = np.linspace(-1, 1, 101)
    trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise
    return [trX, trY]

def inputs_placeholder():    
    X = tf.placeholder("float") # create symbolic variables
    Y = tf.placeholder("float")
    return [X, Y]

def model(X, Y):
    '''
    symbolic graph and ops are defined here
    
    '''
    w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
    y_model = tf.mul(X, w) # lr is just X*w    
    cost = tf.square(Y - y_model) # use square error for cost function    
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    return [train_op, w]

def train(trX, trY, train_op, w):
    ''' train the model '''    
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize variables (in this case just variable W)
        tf.global_variables_initializer().run()
    
        for i in range(100):
            for (x, y) in zip(trX, trY):
                sess.run(train_op, feed_dict={X: x, Y: y})
    
        print(sess.run(w))  # It should be something around 2

if __name__ == '__main__':    
    ''' '''
    # 1 create data
    [trX, trY] = create_data()
    # 2. define the inputs
    [X, Y] = inputs_placeholder()
    # 3. define the model
    [train_op, w] = model(X, Y)
    # 4. training
    train(trX, trY, train_op, w)
    
    
    
    
    