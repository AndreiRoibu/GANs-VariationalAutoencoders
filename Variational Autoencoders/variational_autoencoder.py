import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

NormalDist = tf.contrib.distributions.Normal
BernoulliDist = tf.contrib.distributions.Bernoulli

class DenseLayer(object):
    """Dense Layer class
    """
    def __init__(self, M1, M2, activation = tf.nn.relu):
        """Constructor function
        """
        self.W = tf.Variable(tf.random.normal(shape = (M1, M2)) * np.sqrt(2/(M1+M2)))
        self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.activation = activation

    def forward(self, X):
        """Function performing forward pass
        """
        return self.activation(tf.matmul(X, self.W) + self.b)

class VariationalAutoencoder():
    """ Class defininf the autoencoder
    """
    def __init__(self, D, hidden_layer_sizes):
        """ Constructor function of the autoencoder
        D - input dimensionality
        hidden_layer_sizes - specify the size of every layer in the encoder (decoder is in reverse)
        """

        # First, we create a placeholder for a batch of training data
        self.X = tf.placeholder(tf.float32, shape = (None, D))

        # Then, we create the ENCODER
        self.encoder_layers = []
        M_input = D
        for M_output in hidden_layer_sizes[:-1]:
            layer = DenseLayer(M_input, M_output)
            self.encoder_layers.append(layer)
            M_input = M_output

        # The encoder final layer is unbounded (no activation function)
        # The final layer has 2x outputs (outputs both means and variances)
        M_final_layer = hidden_layer_sizes[-1]
        final_layer = DenseLayer(M_input, M_final_layer * 2, activation= lambda x:x)
        self.encoder_layers.append(final_layer)

        # We then get the mean and std, and pass it through the softplus
        layer_value =  self.X
        for layer in self.encoder_layers:
            layer_value = layer.forward(layer_value)
        self.means = layer_value[:, :M_final_layer]
        self.stddev = tf.nn.softplus(layer_value[:, M_final_layer:]) + 1e-6 # added for smoothing and preventing singularities

        # Finally, we sample Z using a stochastic tensor
        standard_normal = NormalDist(
            loc= self.means,
            scale = self.stddev
        )
        self.Z = standard_normal.sample()

        # This completes the encoder.
        # We now create the DECODER

        self.decoder_layers = []
        M_input = M_final_layer
        for M_output in reversed(hidden_layer_sizes[:-1]):
            layer = DenseLayer(M_input, M_output)
            self.decoder_layers.append(layer)
            M_input = M_output

        # The last layer does not have an activation function, as it outputs a binary probability (Bernoulli)
        layer = DenseLayer(M_input, D, activation=lambda x: x)
        self.decoder_layers.append(layer)

        # The ouput requires logits, which are obtained as such
        layer_value = self.Z
        for layer in self.decoder_layers:
            layer_value = layer.forward(layer_value)
        logits = layer_value

        self.X_hat_distribution = BernoulliDist(logits= logits)

        # We calculate the posterior predictive function.
        # We take a sample from the distribution for calculating the posterior predictive function
        # Claculating the sigmoid aslso produces the mean outout image

        self.posterior_predictive = self.X_hat_distribution.sample()
        self.posterior_predictive_probability = tf.nn.sigmoid(logits)

        # We then caclulate the prior predictive function.
        # We first sample for a Z in N(0,1)
        standard_normal = NormalDist(
            loc = np.zeros(M_final_layer, dtype = np.float32),
            scale = np.ones(M_final_layer, dtype = np.float32)
        )
        Z_std = standard_normal.sample(1)
        layer_value = Z_std
        for layer in self.decoder_layers:
            layer_value = layer.forward(layer_value)
        logits = layer_value
        prior_predictive_distribution = BernoulliDist(logits = logits)
        self.prior_predictive = prior_predictive_distribution.sample()
        self.prior_predictive_probability = tf.nn.sigmoid(logits)

        # Finally, we calculate the prior predictive from an input
        # This is used to generated visualizations
        self.Z_input = tf.placeholder(tf.float32, shape = (None, M_final_layer))
        layer_value = self.Z_input
        for layer in self.decoder_layers:
            layer_value = layer.forward(layer_value)
        logits = layer_value
        self.prior_predictive_from_input_probability = tf.nn.sigmoid(logits)

        # After all has been constructed, we build the cost.
        # The cost is composed of the KL Divergence and the expected log likelihood
        # The KL dirvegnece is between Z and the standard normal previously defined
        # Summing along axis=1, as outpus is NxD and we wish KL per sample.

        KL_divergence = - tf.log(self.stddev) + 0.5 * (np.power(self.stddev, 2) + np.power(self.means, 2)) - 0.5
        KL_divergence = tf.reduce_sum(KL_divergence, axis=1)
        expected_log_likelihood = tf.reduce_sum( self.X_hat_distribution.log_prob(self.X), axis = 1)
        self.elbo = tf.reduce_sum(expected_log_likelihood - KL_divergence)
        self.training_optimizer =tf.train.RMSPropOptimizer(learning_rate = 1e-3).minimize(-self.elbo)

        # Finally, we set up the session and variables:
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

    def fit(self, X, epochs = 30, batch_size = 64):
        """This is the fit function
        """
        costs = []
        number_batches = len(X) // batch_size
        current_directory = os.path.dirname(os.path.realpath(__file__))

        for i in range(epochs):
            print("Epoch {}/{}".format(i, epochs))
            np.random.shuffle(X)
            
            for j in range(number_batches):
                batch = X [ j * batch_size : (j+1) * batch_size ]
                _, cost = self.sess.run( (self.training_optimizer, self.elbo), feed_dict={self.X: batch} )
                cost = cost / batch_size
                costs.append(cost)

                if j % 100 == 0:
                    print("Iteration: {}, cost: {}".format(j, cost))

            save_path = self.saver.save(self.sess, current_directory + "/models/variational_autoencoder.ckpt")
            print("Model saved in path: %s" % save_path)

        plt.plot(costs)
        plt.title("Costs")
        plt.show()

    def posterior_predictive_sample(self, X):
        """ Function calculating the posterior predictive sample p(x_new | x)
        """
        return self.sess.run( self.posterior_predictive, feed_dict = {self.X: X} )

    def prior_predictive_sample_with_probability(self):
        """ Function calculating the sample from p(x_new | z), where z ~ N(0,1)
        """
        return self.sess.run( (self.prior_predictive, self.prior_predictive_probability) )

    def prior_predictive_sample_with_input(self, Z):
        """ Function calculating the prior predictive p(x_new | z)
        """
        return self.sess.run( self.prior_predictive_from_input_probability, feed_dict = {self.Z_input : Z} )

    def transform(self, X):
        """ This function maps an input X into corresponding latent vectors Z
        """
        return self.sess.run(self.means, feed_dict={self.X: X})

def main():
    """ This represents the main function
    """

    X, Y = utils.load_MNIST()
    # Converting inputs to binary variables, which are Bernoulli random variables (not necessary, but improves performance)
    X = (X > 0.5).astype(np.float32) 
    vae = VariationalAutoencoder(784, [200,100])
    vae.fit(X.copy()) # Added copy as fit shuffles the data and we wish to preserve the order

    # To test the network, we plot the reconstruction
    # plot reconstruction
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = vae.posterior_predictive_sample([x]).reshape(28, 28)
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(1,2,2)
        plt.imshow(im, cmap='gray')
        plt.title("Sampled")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True

    # plot output from random samples in latent space
    done = False
    while not done:
        im, probs = vae.prior_predictive_sample_with_probability()
        im = im.reshape(28, 28)
        probs = probs.reshape(28, 28)
        plt.subplot(1,2,1)
        plt.imshow(im, cmap='gray')
        plt.title("Prior predictive sample")
        plt.subplot(1,2,2)
        plt.imshow(probs, cmap='gray')
        plt.title("Prior predictive probs")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True

    # Finally, we want to add some functionality for visualizing the latent space
    # We have a latent space of dimensionality 2 centered at 0
    # These images are a visual representation of what the latent space has learned to encode

    Z = vae.transform(X)
    plt.scatter(Z[:,0], Z[:, 1], c=Y, s=10)
    plt.show()

    # We create a 20x20 grid in range [-3,3] centered on the origin.
    number_images_per_side = 20
    x_values = np.linspace(-3, 3, number_images_per_side)
    y_values = np.linspace(-3, 3, number_images_per_side)
    image = np.empty((28 * number_images_per_side, 28 * number_images_per_side)) # Final image formed of 28x28 MNIST images

    # We do a loop to collect all the Z data points from the grid.
    # This is done in order to call the predict function once.
    Z_points = []
    for i,x in enumerate(x_values):
        for j,y in enumerate(y_values):
            z_point = [x,y]
            Z_points.append(z_point)
    X_reconstructed = vae.prior_predictive_sample_with_input(Z_points)

    # Finally, we place each reconstructed image in its correponding spot
    k = 0
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            x_reconstructed = X_reconstructed[k]
            k += 1
            x_reconstructed = x_reconstructed.reshape(28, 28)
            image[ (number_images_per_side - i - 1) * 28: (number_images_per_side - i) * 28, j*28: (j+1)*28 ] = x_reconstructed

    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
  main()