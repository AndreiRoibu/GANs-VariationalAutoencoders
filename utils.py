import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import os
import imageio
import glob
from natsort import natsorted


def load_MNIST():
    
    current_directory = os.path.dirname(os.path.realpath(__file__))
    train_df = pd.read_csv(current_directory + '/data/MNIST/train.csv')
    train_data = train_df.to_numpy()
    
    Y = train_data[:, 0]
    X = train_data[:, 1:] / 255.0

    X, Y = shuffle(X, Y) # We shuffle the data to break any correlations

    return X, Y

def make_gif(hook):

    anim_file = hook+'_dcgan.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('./samples/'+hook+'_uniform_samples*.png')
        filenames = natsorted(filenames)
        # last = -1
        for i,filename in enumerate(filenames):
            # frame = 2*(i**0.5)
            # print(frame)
            # if round(frame) > round(last):
            #     last = frame
            #     print("---")
            #     print(last)
            #     print(">>>>")
            # else:
            #     continue
            image = imageio.imread(filename)
            writer.append_data(image)


    # import IPython
    #     if IPython.version_info > (6,2,0,''):
    #     display.Image(filename=anim_file)

    

# if __name__ == '__main__':

    # load_MNIST()
    # make_gif('mnist')