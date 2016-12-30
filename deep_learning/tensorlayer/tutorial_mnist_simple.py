#! /usr/bin/python
# -*- coding: utf8 -*-
# modular version


import tensorflow as tf
import tensorlayer as tl



# prepare data
def load_data(): 
    '''
    load mnist data
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = \
                                    tl.files.load_mnist_dataset(shape=(-1,784))
    return [X_train, y_train, X_val, y_val, X_test, y_test]

def input_placeholder(): 
    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
    return [x, y_]

def model(x, y_):                    
    # define the network
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800,
                                    act = tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800,
                                    act = tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    # the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
    # speed up computation, so we use identity here.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    network = tl.layers.DenseLayer(network, n_units=10,
                                    act = tf.identity,
                                    name='output_layer')
    
    # define cost function and metric.
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    
    # define the optimizer
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
    return [network, train_op, cost, acc]


if __name__ =='__main__':
    '''
    
    '''
    sess = tf.InteractiveSession()
    # 1. load data
    [X_train, y_train, X_val, y_val, X_test, y_test] = load_data()
    # 2 define inputs
    [x, y_] = input_placeholder()
    # 3 define the model
    [network, train_op, cost, acc] = model(x, y_)
    # 4. initialize all variables
    sess.run(tf.initialize_all_variables())
    
    # 5. print network information
    network.print_params()
    network.print_layers()
    
    # 6 train the network
    tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                acc=acc, batch_size=500, n_epoch=500, print_freq=5,
                X_val=X_val, y_val=y_val, eval_train=False)
    
    # 7 evaluation
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)
    
    # 8 save the network to .npz file
    tl.files.save_npz(network.all_params , name='model.npz')

    sess.close()
    









