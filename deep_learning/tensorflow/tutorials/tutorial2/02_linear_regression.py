"""Simple tutorial for using TensorFlow to compute a linear regression.

Parag K. Mital, Jan. 2016"""
# %% imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# hyperparameters
n_observations = 100

def create_data(): 
    # %% Let's create some toy data
    plt.ion()    
    fig, ax = plt.subplots(1, 1)
    xs = np.linspace(-3, 3, n_observations)
    ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
    ax.scatter(xs, ys)
    fig.show()
    plt.draw()
    return [xs, ys]

def inputs_placeholder():
    # %% tf.placeholders for the input and output of the network. Placeholders are
    # variables which we need to fill in when we are ready to compute the graph.
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    return [X, Y]

def model(X, Y): 
    # %% We will try to optimize min_(W,b) ||(X*w + b) - y||^2
    # The `Variable()` constructor requires an initial value for the variable,
    # which can be a `Tensor` of any type and shape. The initial value defines the
    # type and shape of the variable. After construction, the type and shape of
    # the variable are fixed. The value can be changed using one of the assign
    # methods.
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    Y_pred = tf.add(tf.multiply(X, W), b)
    
    # %% Loss function will measure the distance between our observations
    # and predictions and average over them.
    cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)
    
    # %% if we wanted to add regularization, we could add other terms to the cost,
    # e.g. ridge regression has a parameter controlling the amount of shrinkage
    # over the norm of activations. the larger the shrinkage, the more robust
    # to collinearity.
    # cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))
    
    # %% Use gradient descent to optimize W,b
    # Performs a single step in the negative gradient
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return [optimizer, cost]
    
def train(xs, ys, X, Y, optimizer, cost):
    '''
    data: xs, ys
    graph: X, Y, optimizer, cost
    '''
    # %% We create a session to use the graph
    n_epochs = 100
    with tf.Session() as sess:
        # Here we tell tensorflow that we want to initialize all
        # the variables in the graph so we can use them
        sess.run(tf.global_variables_initializer())
    
        # Fit all training data
        prev_training_cost = 0.0
        for epoch_i in range(n_epochs):
            for (x, y) in zip(xs, ys):
                sess.run(optimizer, feed_dict={X: x, Y: y})
    
            training_cost = sess.run(
                cost, feed_dict={X: xs, Y: ys})
            print(training_cost)
  
            # Allow the training to quit if we've reached a minimum
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                break
            prev_training_cost = training_cost

if __name__ == '__main__':
    ''' '''
    # create data
    [xs, ys] = create_data()
    # define inputs
    [X, Y] = inputs_placeholder()
    # model
    [optimizer, cost] = model(X, Y)
    # train
    train(xs, ys, X, Y, optimizer, cost)
    



