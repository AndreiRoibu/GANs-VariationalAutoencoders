import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

def load_MNIST():
    
    current_directory = os.path.dirname(os.path.realpath(__file__))
    train_df = pd.read_csv(current_directory + '/data/MNIST/train.csv')
    train_data = train_df.to_numpy()
    
    Y = train_data[:, 0]
    X = train_data[:, 1:] / 255.0

    X, Y = shuffle(X, Y) # We shuffle the data to break any correlations

    return X, Y
    

# if __name__ == '__main__':

#     load_MNIST()