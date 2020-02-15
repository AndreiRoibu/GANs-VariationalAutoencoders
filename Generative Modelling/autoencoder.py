import utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

class Autoencoder:
    """Autoencoder class: 1 hidden layer neural network, ReLU activation function, sigmoid output
    """
    def __init__ (self, input_layer_size, hidden_layer_size):
        """Constructor function
        """
        # First, define the input data
        self.X = tf.placeholder(tf.float32, shape=(None, input_layer_size))

        # Define the input to hidden layer (Xavier initialisation)

        self.W = tf.Variable(tf.random_normal(shape=(input_layer_size, hidden_layer_size)) * np.sqrt(2.0 / (input_layer_size + hidden_layer_size)))
        self.b = tf.Variable(np.zeros(hidden_layer_size).astype(np.float32))

        # Define the hidden layer to output (Xavier initialisation)

        self.W2 = tf.Variable(tf.random_normal(shape=(hidden_layer_size, input_layer_size)) * np.sqrt(2.0 / (input_layer_size + hidden_layer_size)))
        self.b2 = tf.Variable(np.zeros(input_layer_size).astype(np.float32))

        # Generate the output

        self.Z = tf.nn.relu( tf.matmul(self.X, self.W) + self.b )
        self.logits = tf.matmul( self.Z, self.W2 ) + self.b2
        self.X_hat = tf.nn.sigmoid(self.logits)

        # Define the cost function (cross entropy)

        self.cost = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = self.X,
                logits = self.logits
            )
        )

        # Create the trainer

        self.trainer = tf.train.RMSPropOptimizer(learning_rate = 1e-3).minimize(self.cost)

        # Create the session and variables

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def fit(self, X, epochs = 30, batch_size = 64):
        """Function fitting the data
        """

        costs = []
        number_batches = len(X) // batch_size

        current_directory = os.path.dirname(os.path.realpath(__file__))

        for i in range(epochs):

            print("Epoch {}/{}".format(i, epochs))
            
            np.random.shuffle(X)
            
            for j in range(number_batches):
                batch = X [ j * batch_size : (j+1) * batch_size ]

                _, cost = self.sess.run( (self.trainer, self.cost), feed_dict={self.X: batch} )

                cost = cost / batch_size
                costs.append(cost)

                if j % 100 == 0:
                    print("Iteration: {}, cost: {}".format(j, cost))

            save_path = self.saver.save(self.sess, current_directory + "/models/autoencoder.ckpt")
            print("Model saved in path: %s" % save_path)

        plt.plot(costs)
        plt.title("Costs")
        plt.show()

    def predict(self, X):
        """Calculates X_hat given X"
        """
        return self.sess.run(self.X_hat, feed_dict={self.X: X})

def main():
    """ Main function
    """
    X, _ = utils.load_MNIST()

    model = Autoencoder(784, 300)
    model.fit(X)

    # plot reconstruction
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        prediction = model.predict([x]).reshape(28, 28)
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(1,2,2)
        plt.imshow(prediction, cmap='gray')
        plt.title("Reconstruction")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True

if __name__ == '__main__':
    main()