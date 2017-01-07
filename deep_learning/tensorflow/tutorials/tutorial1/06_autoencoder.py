import tensorflow as tf
import numpy as np
import input_data

# hyperparameters
mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 500
corruption_level = 0.3

def inputs_placeholder(): 
    
    # create node for input data
    X = tf.placeholder("float", [None, n_visible], name='X')
    
    # create node for corruption mask
    mask = tf.placeholder("float", [None, n_visible], name='mask')
    return [X, mask]

def autoencoder(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X  # corrupted X

    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input
    return Z


def model(X, mask):
    
    # create nodes for hidden variables
    W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
    W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                               minval=-W_init_max,
                               maxval=W_init_max)
    
    W = tf.Variable(W_init, name='W')
    b = tf.Variable(tf.zeros([n_hidden]), name='b')
    
    W_prime = tf.transpose(W)  # tied weights between encoder and decoder
    b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')
    
    # build model graph
    Z = autoencoder(X, mask, W, b, W_prime, b_prime)
    
    # create cost function
    cost = tf.reduce_sum(tf.pow(X - Z, 2))  # minimize squared error
    train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer
    return [cost, train_op]
    
def load_data(): 
    # load MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return [trX, trY, teX, teY]
    
def train(sess, trX, trY, teX, teY, X, mask, cost, train_op):
    ''' 
    data -- trX, trY, teX, teY
    graph -- (inputs) X, mask, (outputs) cost, train_op
    
    '''
    
    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))


if __name__ == '__main__':
    ''' '''
    # load data
    [trX, trY, teX, teY] = load_data()
    # define inputs
    [X, mask] = inputs_placeholder()
    # define model
    [cost, train_op] = model(X, mask)
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        # train
        train(sess, trX, trY, teX, teY, X, mask, cost, train_op)
    
    
    
    
    
    
    
    
