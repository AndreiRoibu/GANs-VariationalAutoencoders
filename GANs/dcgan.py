import os
import utils
import scipy as sp
import numpy as np
import tensorflow as tf
# import tensorflow-gpu as tf
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import imageio

# First, we define some hyperparameters (borrowed from other research)
learning_rate = 0.0002
beta1 = 0.5
batch_size = 64
epochs = 3
save_sample_period = 50

# Make a samples folde, for saving the output samples
# Script can stall during training, while it waits for user to close plots (GANs take long to train)
if not os.path.exists('samples'):
    os.mkdir('samples')
if not os.path.exists('uniform_samples'):
    os.mkdir('uniform_samples')

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
            shape = (filterSize, filterSize, Min, Mout),
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
            padding='SAME'
        )

        convolution_output = tf.nn.bias_add(convolution_output, self.b)

        if self.apply_batch_norm:
            convolution_output =tf.contrib.layers.batch_norm(
                convolution_output,
                decay = 0.9,
                updates_collections = None,
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
            shape = (filterSize, filterSize, Mout, Min),
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
                updates_collections=None,
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
                updates_collections = None,
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

        # We pass in the sample images through the discriminator to get the sample logits to find the discriminator cost

        with tf.variable_scope("discriminator") as scope:
            scope.reuse_variables()
            sample_logits = self.discriminator_forward(self.sample_images, reuse=True)

        # When in "TEST" mode, we use this code to generate sample images (batch normalisation is different)

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            self.sample_images_test = self.generator_forward(self.Z, reuse=True, is_training=False)

        # This is where the COST functions are built
        # The disciminator cost function:
        #   - is built in 2 steps: 1 for real images, 1 for fake images
        #   - is built using binary cross-entropy
        #   - the total cost is the mean of the 2 cost functions

        self.discriminator_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits = logits,
            labels = tf.ones_like(logits)
        )

        self.discriminator_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits = sample_logits,
            labels = tf.zeros_like(sample_logits)
        )

        self.discriminator_cost = tf.reduce_mean(self.discriminator_cost_real) + tf.reduce_mean(self.discriminator_cost_fake)

        # The generator cost function:
        #   - is the binary cross entropy if the sample logits, \
        #   - the target is set to = 1

        self.generator_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = sample_logits,
                labels = tf.ones_like(sample_logits)
            )
        )

        # We then calculate the accuracy of the discriminator. This is useful for debugging.
        # If the discriminator accuracy is 100%, something is probably wrong!

        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast(sample_logits < 0, tf.float32)
        prediction_number = 2.0 * batch_size
        correct_predictions = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        self.discriminator_accuracy = correct_predictions / prediction_number

        # Now, we create the optimizers
        # One optimizer for each neural network
        # Need to define which params the optimizers need to update

        self.discriminator_parameters = [parameter for parameter in tf.trainable_variables() if parameter.name.startswith('discriminator')]
        self.generator_parameters = [parameter for parameter in tf.trainable_variables() if parameter.name.startswith('generator')]

        self.discriminator_training_operation = tf.train.AdamOptimizer(
            learning_rate,
            beta1 = beta1,
        ).minimize(
            self.discriminator_cost, 
            var_list = self.discriminator_parameters
        )

        self.generator_training_operation = tf.train.AdamOptimizer(
            learning_rate,
            beta1 = beta1
        ).minimize(
            self.generator_cost,
            var_list = self.generator_parameters
        )

        # Finally, we setup the session and variables
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

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
            Min = Min * image_size * image_size
            self.discriminator_denselayers = []
            for Mout, apply_batch_norm in discriminator_sizes['dense_layers']:
                layer_name = "denselayer_%s" % count
                count += 1
                dense_layer = DenseLayer(layer_name, Min, Mout, apply_batch_norm, activation_function= leakyReLU)
                Min = Mout
                self.discriminator_denselayers.append(dense_layer)

            # Build the logistic layer
            layer_name = "denselayer_%s" % count
            self.discriminator_logisticlayer = DenseLayer(layer_name, Min, 1, False, lambda x: x)

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
            # Determine size of data at each step
            # Start at output of generator, and loop backwards to calcualte image dimensions
            # The dimensions array will be backwards (first layer of generator is the last) -> needs reversed
            input_dimensions = [self.image_lenght]
            dimension = self.image_lenght
            for _, _, stride, _ in reversed(generator_sizes['conv_layers']):
                dimension = int(np.ceil(float(dimension) / stride)) 
                input_dimensions.append(dimension)
            input_dimensions = list(reversed(input_dimensions))
            self.generator_dimensions = input_dimensions
    
            # We now go in reverse - first, build the dense layers:
            Min = self.latent_dimension
            self.generator_denselayers = []
            count = 0
            for Mout, apply_batch_norm in generator_sizes['dense_layers']:
                layer_name = "generator_denselayer_%s" % count
                count += 1
                dense_layer = DenseLayer(layer_name, Min, Mout, apply_batch_norm)
                Min = Mout
                self.generator_denselayers.append(dense_layer)

            #Final dense layer
            Mout = generator_sizes['projection'] * input_dimensions[0] * input_dimensions[0]
            layer_name = 'generator_denselayer_%s' % count
            count += 1
            dense_layer = DenseLayer(layer_name, Min, Mout, apply_batch_norm= not generator_sizes['batchNorm_after_projection'])
            self.generator_denselayers.append(dense_layer)

            # Build the fs-conv layers:
            Min = generator_sizes['projection']
            self.generator_convlayers = []
            num_ReLUs = len(generator_sizes["conv_layers"]) - 1
            activation_functions = [tf.nn.relu] * num_ReLUs + [generator_sizes['output_activation']]

            for i in range(len(generator_sizes['conv_layers'])):
                layer_name = "generator_convlayer_%s" % i
                Mout, filter_size, stride, apply_batch_norm = generator_sizes['conv_layers'][i]
                activation_function = activation_functions[i]
                output_shape = [self.batch_size, input_dimensions[i+1], input_dimensions[i+1], Mout]
                
                conv_layer = FractionallyStridedConvLayer(layer_name, Min, Mout, output_shape, apply_batch_norm, filter_size, stride, activation_function)
                self.generator_convlayers.append(conv_layer)
                Min = Mout

            self.generator_sizes = generator_sizes
            return self.generator_forward(Z)

    def generator_forward(self, Z, reuse = None, is_training = True):
        """ This function performs a forward step through the generator
        """
        output = Z
        for layer in self.generator_denselayers:
            output = layer.forward(output, reuse, is_training)

        output = tf.reshape(output, [-1, self.generator_dimensions[0], self.generator_dimensions[0], self.generator_sizes['projection']])

        if self.generator_sizes['batchNorm_after_projection']:
            output = tf.contrib.layers.batch_norm(
                output,
                decay = 0.9,
                updates_collections = None,
                epsilon = 1e-5,
                scale = True,
                is_training = is_training,
                reuse = reuse,
                scope = 'batchNorm_after_projection'
            )

        for layer in self.generator_convlayers:
            output = layer.forward(output, reuse, is_training)

        return output

    def fit(self, X, name):
        """This is the function which fits the model
        When fitting, we do 1 iteration of gradient descent on the discriminator, and 2 iterationg of gradient descenet on the generator
        This is done to prevent the discriminator from learning too fast (it's accuracy increasing too much)
        
        This function also generates and saves samples to disc periodically.
        The code generates 64 samples, and arranges them in an 8x8 grid as 1 image

        Once training is complete, an image of the cost progression is saved
        """
        
        discriminator_costs = []
        generator_costs = []
        number_inputs = len(X)
        number_batches = number_inputs // batch_size
        iters = 0
        current_directory = os.path.dirname(os.path.realpath(__file__))

        Z_uniform = np.random.uniform(-1, 1, size = (64, self.latent_dimension))

        for i in range(epochs):
            print("---------- EPOCH {} ----------".format(i))
            np.random.shuffle(X)
            for j in range(number_batches):
                t0 = datetime.now()

                if type(X[0]) is str:
                    # This means it is a large dataset
                    batch = utils.files2images(
                        X[j * batch_size: (j+1) * batch_size]
                    )
                else:
                    # is MNIST data
                    batch = X[j * batch_size: (j+1) * batch_size]
               
                # We generate a random Z from a uniform distribution
                Z = np.random.uniform(-1, 1, size = (batch_size, self.latent_dimension))

                # 1 iter of GD on discriminator ...
                _, discriminator_cost, discriminator_accuracy = self.sess.run(
                    (self.discriminator_training_operation, self.discriminator_cost, self.discriminator_accuracy),
                    feed_dict = {self.X: batch, self.Z: Z, self.batch_size: batch_size}
                )

                discriminator_costs.append(discriminator_cost)

                #... and 2 iters of GD on the generator
                _, generator_cost1 = self.sess.run(
                    (self.generator_training_operation, self.generator_cost),
                    feed_dict = {self.Z: Z, self.batch_size: batch_size}
                )

                _, generator_cost2 = self.sess.run(
                    (self.generator_training_operation, self.generator_cost),
                    feed_dict = {self.Z: Z, self.batch_size: batch_size}
                )

                generator_costs.append((generator_cost1 + generator_cost2) / 2.0)

                print("batch: %d/%d - dt: %s - disc_acc: %.2f" % (j+1, number_batches, datetime.now() - t0, discriminator_accuracy))

                if j % 100 == 0:
                    save_path = self.saver.save(self.sess, current_directory + "/models/"+name+"/dcgan.ckpt")
                    print("Model saved in path: %s" % save_path)

                # Periodicaly, save images:
                iters += 1
                if iters % save_sample_period == 0:
                    print("Saving a sample...")
                    samples = self.sample(64) # shape is (64, D, D, color channel) - sample function defined below!
                    uniform_samples = self.uniform_sample(64, Z_uniform)
                    dimension = self.image_lenght

                    if samples.shape[-1] == 1: # i.e if color ==1 , we want a 2D image NxN
                        samples = samples.reshape(64, dimension, dimension)
                        uniform_samples = uniform_samples.reshape(64, dimension, dimension)
                        flat_image = np.empty((8*dimension, 8*dimension))
                        flat_uniform_image = np.empty((8*dimension, 8*dimension))

                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i*dimension:(i+1)*dimension, j*dimension:(j+1)*dimension] = samples[k].reshape(dimension, dimension)
                                flat_uniform_image[i*dimension:(i+1)*dimension, j*dimension:(j+1)*dimension] = uniform_samples[k].reshape(dimension, dimension)
                                k += 1
                    else:
                        flat_image = np.empty((8*dimension, 8*dimension, 3)) # we want an image with 3 color channels
                        flat_uniform_image = np.empty((8*dimension, 8*dimension, 3)) # we want an image with 3 color channels
                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i*dimension:(i+1)*dimension, j*dimension:(j+1)*dimension] = samples[k]
                                flat_uniform_image[i*dimension:(i+1)*dimension, j*dimension:(j+1)*dimension] = uniform_samples[k]
                                k += 1


                    sp.misc.imsave(
                        'samples/'+name+'_random_samples_at_iter_%d.png' % iters,
                        flat_image,
                    )

                    sp.misc.imsave(
                        'uniform_samples/'+name+'_uniform_samples_at_iter_%d.png' % iters,
                        flat_uniform_image,
                    )


        save_path = self.saver.save(self.sess, current_directory + "/models/"+name+"/dcgan.ckpt")
        save_path = self.saver.save(self.sess, current_directory + "/models/dcgan.ckpt")
        print("Model saved in path: %s" % save_path)
        
        plt.clf()
        plt.plot(discriminator_costs, label = 'Discriminator Cost')
        plt.plot(generator_costs, label= 'Generator Cost')
        plt.legend()
        plt.title("Training Costs for "+name)
        plt.xlabel('Iterations')
        plt.ylabel('Costs')
        plt.savefig(name+'_cost_vs_iteration.png')

    def sample(self, n):
        """ This function runs the sample_image_test function and generates samples
        """
        Z = np.random.uniform(-1, 1, size = (n, self.latent_dimension))
        samples = self.sess.run(self.sample_images_test, feed_dict = {self.Z: Z, self.batch_size: n})
        return samples

    def uniform_sample(self, n, Z_uniform):
        """ This function runs the sample_image_test function and generates samples
        However, this sample keeps the seed constant, meaning that the same images are generated each time
        """
        uniform_samples = self.sess.run(self.sample_images_test, feed_dict = {self.Z: Z_uniform, self.batch_size: n})
        return uniform_samples


def mnist():
    """ Function that loads MNIST, reshapes it to TF desired input (hight, width, color)
    Then, the function defines the shape of the discriminator and generator
    """

    X, _ = utils.load_MNIST()
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
        'dense_layers': [(1024, True)],
        'output_activation': tf.sigmoid,
    }

    # Create the DCGAN and fit it to the images
    name = 'MNIST'
    GAN = DCGAN(dimensions, colors, discriminator_sizes, generator_sizes)
    GAN.fit(X, name)

    utils.make_gif(name)

def celeb():
    """ Function that loads Celeb data, reshapes it to TF desired input (hight, width, color)
    Then, the function defines the shape of the discriminator and generator
    """

    # This just gets a list of filenames to be loaded in dynamically due to their large number and size
    X = utils.load_Celeb()

    dimensions = 64 # Assumes images are square - uses only 1 dimension
    colors = 3

    # Hyperparamters gathered from other official implementations that worked! Selected with hyper param optimisation techniques

    # Hyperparameter keys: 
    # conv layer: (feature maps, filter size, stride=2, batch norm used?)
    # dense layer: (hidden units, batch norm used?)
    discriminator_sizes = {
        'conv_layers': [(64, 5, 2, False), (128, 5, 2, True), (256, 5, 2, True), (512, 5, 2, True)],
        'dense_layers': []
    }

    # Hyperparameter keys: 
    # z : latent variable dimensionality (drawing uniform random samples from it)
    # projection: initial number of feature maps (flat vector -> 3D image!)
    # batchNorm_after_projection: flag, showing, if we want to use batchnorm after projecting the flat vector
    # conv layer: (feature maps, filter size, stride=2, batch norm used?)
    # dense layer: (hidden units, batch norm used?)
    # output_action: activation function - using sigmoid since the Celeb data is scaled between {-1, 1} - This is recommended by GAN researchers 
    generator_sizes = {
        'z' : 100,
        'projection' : 512,
        'batchNorm_after_projection': True,
        'conv_layers': [(256, 5, 2, True), (128, 5, 2, True), (64, 5, 2, True), (colors, 5, 2, False)],
        'dense_layers': [],
        'output_activation': tf.tanh,
    }

    # Create the DCGAN and fit it to the images
    name = 'Celeb'
    GAN = DCGAN(dimensions, colors, discriminator_sizes, generator_sizes)
    GAN.fit(X, name)

    utils.make_gif(name)


if __name__ == '__main__':
    # mnist()
    celeb()
    
