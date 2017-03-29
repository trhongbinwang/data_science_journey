#!/usr/bin/env python
''' 
how to use tensorboard
1. add operators in model definition
2. add writer in train func
3. run 'tensorboard --logdir=./logs/nn_logs'

'''
import tensorflow as tf
import input_data


def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

# This network is the same as the previous one except with an extra hidden layer + dropout
def mlp_dropout(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o)

def load_data():
    ''' '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return [trX, trY, teX, teY]
    
def inputs_placeholder():
    ''' '''
    X = tf.placeholder("float", [None, 784], name="X")
    Y = tf.placeholder("float", [None, 10], name="Y")
    p_keep_input = tf.placeholder("float", name="p_keep_input")
    p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")
    return [X, Y, p_keep_input, p_keep_hidden]
    
def model(X, Y, p_keep_input, p_keep_hidden):
    w_h = init_weights([784, 625], "w_h")
    w_h2 = init_weights([625, 625], "w_h2")
    w_o = init_weights([625, 10], "w_o")
    
    # Add histogram summaries for weights
    tf.summary.histogram("w_h_summ", w_h)
    tf.summary.histogram("w_h2_summ", w_h2)
    tf.summary.histogram("w_o_summ", w_o)
    
    py_x = mlp_dropout(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
    
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
        # Add scalar summary for cost
        tf.summary.scalar("cost", cost)
    
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
        acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
        # Add scalar summary for accuracy
        tf.summary.scalar("accuracy", acc_op)
    merged = tf.summary.merge_all()
    
    return [train_op, acc_op, merged]

def train(sess, trX, trY, teX, teY, X, Y, p_keep_input, p_keep_hidden, train_op, acc_op, merged):
    '''
    data: trX, trY, teX, teY
    graph: (inputs) X, Y, p_keep_input, p_keep_hidden, (outputs) train_op, acc_op, merged
    
    '''
    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # for 0.8
    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY,
                                          p_keep_input: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)  # Write summary
        print(i, acc)                   # Report the accuracy

if __name__ == '__main__':
    ''' '''
    # load_data
    [trX, trY, teX, teY] = load_data()
    # inputs
    [X, Y, p_keep_input, p_keep_hidden] = inputs_placeholder()
    # model
    [train_op, acc_op, merged] = model(X, Y, p_keep_input, p_keep_hidden)
    
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        # train
        train(sess, trX, trY, teX, teY, X, Y, p_keep_input, p_keep_hidden, train_op, acc_op, merged)
    
    
    
    
    
    
    
    
    
    
