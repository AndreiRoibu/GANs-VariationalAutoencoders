import utils
import numpy as np
import matplotlib.pyplot as plt

from variational_autoencoder import VariationalAutoencoder

def main():
    """ This represents the main function
    """

    X, Y = utils.load_MNIST()
    # Converting inputs to binary variables, which are Bernoulli random variables (not necessary, but improves performance)
    X = (X > 0.5).astype(np.float32) 
    vae = VariationalAutoencoder(784, [200,100,50,10,2])
    vae.fit(X.copy()) # Added copy as fit shuffles the data and we wish to preserve the order

    # We want to add some functionality for visualizing the latent space
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