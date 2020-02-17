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
    def __init__(self, layer_name, Min, Mout, apply_batch_norm, filterSize = 5, stride = 2, activation_function = tf.nn.relu):
        """Constructor class for the Conv layer
        layer_name = layer layer_name
        Min = number of input feature maps
        Mout = number of output feature maps
        apply_batch_norm = flag if to apply batch norm
        """
        self.W = tf.get_variable(
            "W_%s" % layer_name,
            shape = (filterSize, Min, Mout),
            # initializer=tf.contrib.layers.xavier_initializer(),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s" % layer_name,
            shape = (Mout, ),
            initializer= tf.zeros_initializer()
        )

        self.layer_name = layer_name
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
                scope = self.layer_name
            )
        
        return self.activation_function(convolution_output)

class FractionallyStridedConvLayer:
    """ Class defining a fractionally strided convolution layer
    """ 
    def __init__(self, layer_name, Min, Mout, output_shape, apply_batch_norm, filterSize = 5, stride = 2, activation_function = tf.nn.relu):
        """Constructor class for the Conv layer
        layer_name = layer layer_name
        Min = number of input feature maps
        Mout = number of output feature maps
        apply_batch_norm = flag if to apply batch norm
        output_shape = required as input to Conv2Dtranspose

        Note: Input and Output feature maps are switched around in the convolutional filter.
        """

        self.W = tf.get_variable(
            "W_%s" % layer_name,
            shape = (filterSize, Min, Mout),
            # initializer=tf.contrib.layers.xavier_initializer(),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s" % layer_name,
            shape = (Mout, ),
            initializer= tf.zeros_initializer()
        )

        self.layer_name = layer_name
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
            output_shape = self.output_shape,
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
                scope = self.layer_name
            )
        
        return self.activation_function(convolution_output)        

class DenseLayer(object):
    """ This represents the Dense Layer class
    """
    def __init__(self, layer_name, M1, M2, apply_batch_norm, activation_function = tf.nn.relu):
        """ Constructor function
        """
        self.W = tf.get_variable(
            "W_%s" % layer_name,
            shape = (M1, M2),
            # initializer=tf.contrib.layers.xavier_initializer(),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s" % layer_name,
            shape = (M2, ),
            initializer= tf.zeros_initializer()
        )

        self.activation_function = activation_function
        self.layer_name = layer_name
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        """ Function going forward through the layer
        X = input tensor
        reuse = reuse flag
        is_training = flag showing is this is a training or testing operation
        """
        a = tf.matmul(X, self.W) + self.b

        if self.apply_batch_norm:
            a =tf.contrib.layers.batch_norm(
                a,
                decay = 0.9,
                upstate_collections = None,
                epsilon = 1e-5,
                scale = True, 
                is_training = is_training,
                reuse = reuse,
                scope = self.layer_name
            )

        return self.activation_function(a)

class DCGAN:
    """This represents the DCGAN class
    """
    def __init__(self, image_lenght, number_colors, discriminator_sizes, generator_sizes):
        """ This is the constructor, where most of the work in this class will be based
        """

        # First, save the inputs for later
        self.image_lenght = image_lenght
        self.number_colors = number_colors
        self.latent_dimension = generator_sizes['z']

        # Define the input data (assume square images)
        self.X = tf.placeholder(
            tf.float32,
            shape = (None, image_lenght, image_lenght, number_colors),
            name = 'X'
        )

        self.Z = tf.placeholder(
            tf.float32,
            shape = (None, self.latent_dimension),
            name = 'Z'
        )

        self.batch_size = tf.placeholder(
            tf.int32,
            shape = (),
            name = 'batch_size'
        )

        # After building the inputs, we build the discrimator and the generator

        logits = self.build_discriminator(self.X, discriminator_sizes)

        self.sample_images = self.build_generator(self.Z, generator_sizes)

    # ===========
    # This is where we construct the helper functions

    def build_discriminator(self, X, discriminator_sizes):
        """ This function constructs the discriminator network
        """
        with tf.variable_scope('discriminator') as scope:
            # Build the conv layers:
            self.discriminator_convlayers = []
            Min = self.number_colors
            image_size = self.image_lenght
            count = 0
            for Mout, filter_size, stride, apply_batch_norm in discriminator_sizes['conv_layers']:
                layer_name = "convlayer_%s" % count
                count += 1
                conv_layer = ConvLayer(layer_name, Min, Mout, apply_batch_norm, filter_size, stride, activation_function=leakyReLU)
                self.discriminator_convlayers.append(conv_layer)
                Min = Mout
                image_size = int(np.ceil(float(image_size) / stride))

            # Build the dense layers:
            Min_dense_layer = Min * image_size * image_size
            self.discriminator_denselayers = []
            for Mout, apply_batch_norm in discriminator_sizes['dense_layers']:
                layer_name = "denselayer_%s" % count
                count += 1
                dense_layer = DenseLayer(layer_name, Min_dense_layer, Mout, apply_batch_norm, activation_function= leakyReLU)
                Min_dense_layer = Mout
                self.discriminator_denselayers.append(dense_layer)

            # Build the logistic layer
            layer_name = "denselayer_%s" % count
            self.discriminator_logisticlayer = DenseLayer(layer_name, Min_dense_layer, 1, False, lambda x: x)

            # Get the logits:
            logits = self.discriminator_forward(X)

            return logits
        
    def discriminator_forward(self, X, reuse = None, is_training = True):
        """ This function performs a forward step through the discriminator
        """
        discriminator_output = X
        for layer in self.discriminator_convlayers:
            discriminator_output = layer.forward(discriminator_output, reuse, is_training)
        discriminator_output = tf.contrib.layers.flatten(discriminator_output)

        for layer in self.discriminator_denselayers:
            discriminator_output = layer.forward(discriminator_output, reuse, is_training)
        logits = self.discriminator_logisticlayer.forward(discriminator_output, reuse, is_training)

        return logits
    
    def build_generator(self, Z, generator_sizes):
        """ This function constructs the generator network
        """
        with tf.variable_scope("generator") as scope:

            



            # Build the conv layers:
            self.discriminator_convlayers = []
            Min = self.number_colors
            image_size = self.image_lenght
            count = 0
            for Mout, filter_size, stride, apply_batch_norm in discriminator_sizes['conv_layers']:
                layer_name = "convlayer_%s" % count
                count += 1
                conv_layer = ConvLayer(layer_name, Min, Mout, apply_batch_norm, filter_size, stride, activation_function=leakyReLU)
                self.discriminator_convlayers.append(conv_layer)
                Min = Mout
                image_size = int(np.ceil(float(image_size) / stride))

            # Build the dense layers:
            Min_dense_layer = Min * image_size * image_size
            self.discriminator_denselayers = []
            for Mout, apply_batch_norm in discriminator_sizes['dense_layers']:
                layer_name = "denselayer_%s" % count
                count += 1
                dense_layer = DenseLayer(layer_name, Min_dense_layer, Mout, apply_batch_norm, activation_function= leakyReLU)
                Min_dense_layer = Mout
                self.discriminator_denselayers.append(dense_layer)

            # Build the logistic layer
            layer_name = "denselayer_%s" % count
            self.discriminator_logisticlayer = DenseLayer(layer_name, Min_dense_layer, 1, False, lambda x: x)

            # Get the logits:
            logits = self.discriminator_forward(X)

            return logits



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