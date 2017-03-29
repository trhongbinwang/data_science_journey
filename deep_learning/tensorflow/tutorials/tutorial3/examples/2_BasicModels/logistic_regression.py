'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1


def load_data():
    # Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    return mnist

def inputs_placeholder():
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
    return [x, y]
    
def model(x, y):
    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), axis=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return [cost, optimizer, accuracy]
    
def training(sess, mnist, x, y, cost, optimizer, accuracy):
    ''' 
    data: mnist
    graph: x, y, cost, optimizer, accuracy
    
    '''
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    ''' '''
    # load data
    mnist = load_data()
    # define inputs
    [x, y] = inputs_placeholder()
    # model
    [cost, optimizer, accuracy] = model(x, y)
    # Launch the graph
    with tf.Session() as sess:
        # Initializing the variables
        init = tf.global_variables_initializer()
    
        sess.run(init)
        # training
        training(sess, mnist, x, y, cost, optimizer, accuracy)
    
    
    
    
    
    
    
    
    
    





