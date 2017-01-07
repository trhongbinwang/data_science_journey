"""Tutorial on how to create a denoising autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
from libs.utils import corrupt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt


# hyperparameters
dimensions=dimensions=[784, 256, 64]
learning_rate = 0.001
batch_size = 50
n_epochs = 10


def load_data():
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    return [mnist, mean_img]
    
def inputs_placeholder():
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    corrupt_prob = tf.placeholder(tf.float32, [1])
    return [x, corrupt_prob]
    
def model(x, corrupt_prob):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """

    # input to the network
    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic.  We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in.  But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return [cost, optimizer, y]
    

def train(sess, mnist, mean_img,  x, corrupt_prob, cost, optimizer):
    '''
    sess
    data: mnist, mean_img
    graph: x, corrupt_prob, cost, optimizer
    
    '''
    # %%

    # %%
    # Fit all training data
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={
                x: train, corrupt_prob: [1.0]})
        print(epoch_i, sess.run(cost, feed_dict={
            x: train, corrupt_prob: [1.0]}))

def evaluate(sess, mnist, x, corrupt_prob, y):
    '''
    sess
    data: mnist
    graph:x, corrupt_prob, y
    '''
    # %%
    # Plot example reconstructions
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(y, feed_dict={
        x: test_xs_norm, corrupt_prob: [0.0]})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    fig.show()
    plt.draw()

if __name__ == '__main__':
    ''' '''
    # load data
    [mnist, mean_img] = load_data()
    # define inputs    
    [x, corrupt_prob] = inputs_placeholder()
    # define model
    [cost, optimizer, y] = model(x, corrupt_prob)
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # training the model
    train(sess, mnist, mean_img,  x, corrupt_prob, cost, optimizer)
    # evaluate
    evaluate(sess, mnist, x, corrupt_prob, y)

    
    
    
    
    
    
    
    
    
    
    
