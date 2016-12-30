'''This script demonstrates how to build a variational autoencoder with Keras.

modified in modular way
the principle is to put the details into functions and pipeline consists of similiar level of funcs
how to reuse the existing trained layers

2 tips:
1. all models in one function to enable share and reuse
2. define the loss function inside the model function to get extra inputs. 

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


# hyperparameter
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 10
epsilon_std = 1.0


def sampling(args):
    '''
    sample a data point from a Gaussian dist
    '''
    z_mean, z_log_var = args
    # when using low level operator, import K
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


#def vae_loss(x, x_decoded_mean):
#    '''
#    vae_loss = reconstruction term  # no extra inputs
     
#    '''
#    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
#    return xent_loss


def model():
    '''
    build all models at one place
    In this way, all shared layers reused
    '''
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    vae = Model(x, x_decoded_mean)
    
    def vae_loss(x, x_decoded_mean):
        '''
        vae_loss = reconstruction term + regularizer term
        define the loss inside the model is unusual. this is because kl_loss
        rely on extra inputs such as z_log_var, z_mean. The standard loss func 
        params only accept y_true and y_predict. extra inputs can only get from
        local global.         
        '''
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    
    # encoder
    # build a model to project inputs on the latent space

    encoder = Model(x, z_mean)
    
    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    return [vae, encoder, generator]


def load_data(): 
    '''
    load mnist data
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('original x_train and x_test shape')
    print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
    x_train = x_train.astype('float32') / 255. # normalize
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print('final x_train and x_test shape')
    print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

    return [x_train, y_train, x_test, y_test]



def display_latent_space(encoder, x_test, y_test):
    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    ''' 
    # train the VAE on MNIST digits
    '''
    # 1. load data
    [x_train, y_train, x_test, y_test] = load_data()
    # define the model
    [vae, encoder] = model()
    # train the model
    vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))
    
    # display a 2D plot of the digit classes in the latent space
    display_latent_space(encoder, x_test, y_test)










