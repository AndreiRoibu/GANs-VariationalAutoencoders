import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

def truncate_sample(x):
    """This function ensures that the obtained sample is between (0,1)
    """

    x = np.minimum(x, 1)
    x = np.maximum(x, 0)

    return x

class BayesClassifier:

    # No constructor is required

    def fit(self, X, Y):
        """Function fitting the gaussian
        X - inputs
        Y - classes
        """

        self.classes = len(set(Y)) # We assume classes are in (0...K-1)
        self.gaussians = []
        self.p_y = np.zeros(self.classes) # p(y)

        for individual_class in range(self.classes):

            X_class = X[ Y == individual_class ]

            self.p_y[individual_class] = len(X_class)

            mean = np.mean(X_class, axis=0)
            covariance = np.cov(X_class.T)
            gaussian = {'mean': mean, 'covariance': covariance}
            self.gaussians.append(gaussian)

        self.p_y = self.p_y / self.p_y.sum() # This normalizes p(y)

    def sample_given_y(self, y):
        """ Function drawing a random sample and calculating  p(x | y)
        Finds a gaussian corresponding to y and generates a sample
        """

        gaussian = self.gaussians[y]
        return truncate_sample( mvn.rvs(mean = gaussian['mean'], cov=gaussian['covariance']) )

    def sample(self):
        """ Function choosing a random class y and returns a sample
        """

        y = np.random.choice(self.classes, p= self.p_y)
        return truncate_sample( self.sample_given_y(y) )


if __name__ == '__main__':

    X, Y = utils.load_MNIST()

    classifier = BayesClassifier()
    classifier.fit(X, Y)

    for individual_class in range(classifier.classes):
        sample = classifier.sample_given_y(individual_class).reshape(28, 28)
        mean = classifier.gaussians[individual_class]['mean'].reshape(28, 28)

        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()

    random_sample = classifier.sample().reshape(28,28)
    plt.imshow(random_sample, cmap='gray')
    plt.title("Random Sample from Random Class")
    plt.show()
