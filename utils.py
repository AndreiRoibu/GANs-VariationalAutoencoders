import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import os
import imageio
import glob
from natsort import natsorted
import requests
from tqdm import tqdm
from scipy.misc import imread, imsave, imresize
import zipfile


def load_MNIST():
    
    current_directory = os.path.dirname(os.path.realpath(__file__))
    train_df = pd.read_csv(current_directory + '/data/MNIST/train.csv')
    train_data = train_df.to_numpy()
    
    Y = train_data[:, 0]
    X = train_data[:, 1:] / 255.0

    X, Y = shuffle(X, Y) # We shuffle the data to break any correlations

    return X, Y

def make_gif(hook):
    """ Function which creates a gif from the outputed images
    """
    anim_file = hook+'_dcgan2.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('./uniform_samples/'+hook+'_uniform_samples*.png')
        filenames = natsorted(filenames)
        # last = -1
        for i,filename in enumerate(filenames):
            # frame = 2*(i**0.5)
            # if round(frame) > round(last):
            #     last = frame
            # else:
            #     continue
            image = imageio.imread(filename)
            writer.append_data(image)

def files2images(filenames):
    """ Function which reads images in a file and returns them as data
    """
    return [scaled_images(imread(filename)) for filename in filenames]

def scaled_images(image):
    """ Scales an image between [-1, 1]
    """
    return (image / 255.0) * 2.0 - 1.0

def crop_and_save(inputFile, outputDirectory):
    """ Function which croppes and saves the images
    The function assumes that the middle 108 pixels of the image contain the face
    """
    image = imread(inputFile)
    heigth, width, _ = image.shape
    edge_heigth = int(round((heigth - 108)/2))
    edge_width = int(round((width - 108)/2))
    cropped = image[edge_heigth:(edge_heigth+108), edge_width:(edge_width+108)]
    small = imresize(cropped, (64, 64))

    filename = inputFile.split('/')[-1]
    imsave("%s/%s" % (outputDirectory, filename), small)

def load_Celeb():
    """ Function which dowloads and loads the Celeb database
    Link to manually downloadable data: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
    """
    current_directory = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(current_directory + '/data'):
        os.mkdir(current_directory + '/data')

    # if the data does not exist, we download and prepare it:
    if not os.path.exists(current_directory+'/data/img_align_celeba-cropped'):
        if not os.path.exists(current_directory+'/data/img_align_celeba'):
            print("Extracting the Celeb Data: img_align_celeba.zip")
            with zipfile.ZipFile(current_directory+'/data/img_align_celeba.zip') as zf:
                zip_directory = zf.namelist()[0]
                zf.extractall(current_directory+'/data')
            print("Done!")
        filenames = glob.glob(current_directory+'/data/img_align_celeba/*jpg')
        number_files = len(filenames)
        print("Found {} files!".format(number_files))

        os.mkdir(current_directory+'/data/img_align_celeba-cropped')
        print("Cropping images to standard size")
        for i in range(number_files):
            crop_and_save(filenames[i], current_directory+'/data/img_align_celeba-cropped')
            if i % 1000 == 0:
                print("{} / {}".format(i, number_files))
    
    filenames = glob.glob(current_directory+'/data/img_align_celeba-cropped/*jpg')
    return filenames


if __name__ == '__main__':

    # load_MNIST()
    # make_gif('MNIST')
    X = load_Celeb()
    print(X)
    current_directory = os.path.dirname(os.path.realpath(__file__))
