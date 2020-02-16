import os
import utils
import scipy as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

# First, we define some hyperparameters (borrowed from other research)
learning_rate = 0.0002
beta1 = 0.5
batch_size = 64
epochs = 2
save_sample_period = 50

# Make a samples folde, for saving the output samples
# Script can stall during training, while it waits for user to close plots (GANs take long to train)
if not os.path.exists('samples'):
    os.mkdir('samples')

def leakyReLU(x, alpha=0.2):
    """Function defining the leaky relu operation\
    """
    return tf.maximum(alpha * x, x)

class ConvLayer:
    """ Class defining a convolutional layer
    """
    def __init__(self, name, Min, Mout, apply_batch_norm, filterSize = 5, stride = 2, activation_function = tf.nn.relu):
        """Constructor class for the Conv layer
        name = layer name
        Min = number of input feature maps
        Mout = number of output feature maps
        apply_batch_norm = flag if to apply batch norm
        """
        self.W = tf.get_variable(
            "W_%s" % name,
            shape = (filterSize, Min, Mout),
            # initializer=tf.contrib.layers.xavier_initializer(),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s" % name,
            shape = (Mout, ),
            initializer= tf.zero_initializer()
        )

        self.name = name
        self.activation_function = activation_function
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        """ Function going forward through the layer
        X = input tensor
        reuse = reuse flag
        is_training = flag showing is this is a training or testing operation
        """
        convolution_output = tf.nn.conv2d(
            X,
            self.W,
            strides = [1, self.stride, self.stride, 1],
            padding = 'SAME'
        )

        convolution_output = tf.nn.bias_add(convolution_output, self.b)

        if self.apply_batch_norm:
            convolution_output =tf.contrib.layers.batch_norm(
                convolution_output,
                decay = 0.9,
                upstate_collections = None,
                epsilon = 1e-5,
                scale = True, 
                is_training = is_training,
                reuse = reuse,
                scope = self.name
            )
        
        return self.activation_function(convolution_output)

class FractionallyStridedConvLayer:
    """ Class defining a fractionally strided convolution layer
    """ 
    def __init__(self, name, Min, Mout, output_shape, apply_batch_norm, filterSize = 5, stride = 2, activation_function = tf.nn.relu):
        """Constructor class for the Conv layer
        name = layer name
        Min = number of input feature maps
        Mout = number of output feature maps
        apply_batch_norm = flag if to apply batch norm
        output_shape = required as input to Conv2Dtranspose

        Note: Input and Output feature maps are switched around in the convolutional filter.
        """

        self.W = tf.get_variable(
            "W_%s" % name,
            shape = (filterSize, Min, Mout),
            # initializer=tf.contrib.layers.xavier_initializer(),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s" % name,
            shape = (Mout, ),
            initializer= tf.zero_initializer()
        )

        self.name = name
        self.activation_function = activation_function
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b] 
        self.output_shape = output_shape

    def forward(self, X, reuse, is_training):
        """ Function going forward through the layer
        X = input tensor
        reuse = reuse flag
        is_training = flag showing is this is a training or testing operation
        """
        convolution_output = tf.nn.conv2d_transpose(
            value = X,
            filter = self.W,
            output_shape= = self.output_shape
            strides = [1, self.stride, self.stride, 1],
        )

        convolution_output = tf.nn.bias_add(convolution_output, self.b)

        if self.apply_batch_norm:
            convolution_output =tf.contrib.layers.batch_norm(
                convolution_output,
                decay = 0.9,
                upstate_collections = None,
                epsilon = 1e-5,
                scale = True, 
                is_training = is_training,
                reuse = reuse,
                scope = self.name
            )
        
        return self.activation_function(convolution_output)        

class DenseLayer(object):
    """ This represents the Dense Layer class
    """
    def __init__(self, name, M1, M2, apply_batch_norm, activation_function = tf.nn.relu):
        """ Constructor function
        """
        self.W = tf.get_variable(
            "W_%s" % name,
            shape = (filterSize, Min, Mout),
            # initializer=tf.contrib.layers.xavier_initializer(),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s" % name,
            shape = (Mout, ),
            initializer= tf.zero_initializer()
        )

        self.activation_function = activation_function
        self.name = name
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        """ Function going forward through the layer
        X = input tensor
        reuse = reuse flag
        is_training = flag showing is this is a training or testing operation
        """
        a = tf.matmult(X, self.W) + self.b

        if self.apply_batch_norm:
            a =tf.contrib.layers.batch_norm(
                a,
                decay = 0.9,
                upstate_collections = None,
                epsilon = 1e-5,
                scale = True, 
                is_training = is_training,
                reuse = reuse,
                scope = self.name
            )

        return self.activation_function(a)

class DCGAN:
    """This represents the DCGAN class
    """"
    def __init__():

def mnist():
    """ Function that loads MNIST, reshapes it to TF desired input (hight, width, color)
    Then, the function defines the shape of the discriminator and generator
    """

    X, Y = utils.load_MNIST()
    X = X.reshape(len(X), 28, 28, 1)
    dimensions = X.shape[1] # Assumes images are square - uses only 1 dimension
    colors = X.shape[-1]

    # Hyperparamters gathered from other official implementations that worked! Selected with hyper param optimisation techniques

    # Hyperparameter keys: 
    # conv layer: (feature maps, filter size, stride=2, batch norm used?)
    # dense layer: (hidden units, batch norm used?)
    discriminator_sizes = {
        'conv_layers': [(2, 5, 2, False), (64, 5, 2, True)],
        'dense_layers': [(1024, True)]
    }

    # Hyperparameter keys: 
    # z : latent variable dimensionality (drawing uniform random samples from it)
    # projection: initial number of feature maps (flat vector -> 3D image!)
    # batchNorm_after_projection: flag, showing, if we want to use batchnorm after projecting the flat vector
    # conv layer: (feature maps, filter size, stride=2, batch norm used?)
    # dense layer: (hidden units, batch norm used?)
    # output_action: activation function - using sigmoid since MNIST varies between {0, 1}
    generator_sizes = {
        'z' : 100,
        'projection' : 128,
        'batchNorm_after_projection': False,
        'conv_layers': [(128, 5, 2, True), (colors, 5, 2, False)],
        'output_activation': tf.sigmoid,
    }

    # Create the DCGAN and fit it to the images
    GAN = DCGAN(dimensions, colors, discriminator_sizes, generator_sizes)
    GAN.fit(X)
    # samples = GAN.sample(1)

if __name__ == '__main__':
    mnist()