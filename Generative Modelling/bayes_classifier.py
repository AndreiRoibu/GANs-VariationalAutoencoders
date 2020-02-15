import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import BayesianGaussianMixture

def truncate_sample(x):
    """This function ensures that the obtained sample is between (0,1)
    """

    x = np.minimum(x, 1)
    x = np.maximum(x, 0)

    return x

class BayesClassifierGaussian:

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

class BayesClassifierGMM():

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

            print("Fitting GMM for the %s class" %individual_class)

            X_class = X[ Y == individual_class ]

            self.p_y[individual_class] = len(X_class)

            # Each Gaussian is a Bayesian Gaussian Mixture Object
            # The 10 argument is the maximum number of clusters (chosen arbitrarily, could be more)

            GMM = BayesianGaussianMixture(10)

            # The fit function performs the variational inferance update (could take long, iterative algorithm)

            GMM.fit(X_class)

            self.gaussians.append(GMM)

            print("Finished fitting the GMM for the %s class" %individual_class)
            print("======================================================")

        self.p_y = self.p_y / self.p_y.sum() # This normalizes p(y)

    def sample_given_y(self, y):
        """ Function drawing a random sample and calculating  p(x | y)
        Finds a gaussian corresponding to y and generates a sample
        """

        gaussian_gmm = self.gaussians[y]

        sample = gaussian_gmm.sample()

        # The .sample() function returns a touple (the sample, the cluster)

        cluster_mean = gaussian_gmm.means_[sample[1]]

        return truncate_sample( sample[0].reshape(28,28) ), cluster_mean.reshape(28,28)

    def sample(self):
        """ Function choosing a random class y and returns a sample
        """

        y = np.random.choice(self.classes, p= self.p_y)
        return truncate_sample( self.sample_given_y(y) )

if __name__ == '__main__':

    X, Y = utils.load_MNIST()

    # As 2 classifier methods are employeed, uncomment the desired one.
    
    # classifier = BayesClassifierGaussian()
    classifier = BayesClassifierGMM()


    classifier.fit(X, Y)

    for individual_class in range(classifier.classes):
        if classifier == BayesClassifierGaussian():

            sample = classifier.sample_given_y(individual_class).reshape(28, 28)
            mean = classifier.gaussians[individual_class]['mean'].reshape(28, 28)

        else:

            sample, mean = classifier.sample_given_y(individual_class)
            
        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()

    if classifier == BayesClassifierGaussian():

        random_sample = classifier.sample().reshape(28,28)

    else:

        random_sample = classifier.sample()

    plt.imshow(random_sample, cmap='gray')
    plt.title("Random Sample from Random Class")
    plt.show()
