from keras.datasets import mnist
import scipy.io as sio
import random
import itertools
import numpy as np

def gen_data_MNIST():
    """
    Generation of MNIST Handwritten Digit Data Set

    This data set contains grey images of scanned handwritten digits.
    5000 images are randomly selected for each digit from the training set 
    to create the matrix X. The vector y is randomly selected from the images
    of the test set.

    Returns
    -------  
    X : np.array of shape (784, 50000)
        The 50 000 flattened images of digits.
    y : np.array of shape (50 000)
        The 50 000 labels of images
    """
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    ind=[[]]*10
    for i in range(10):
        ind[i]=random.sample([j for j in range(np.shape(train_X)[0]) if train_y[j]==i],5000)
    ind=list(itertools.chain.from_iterable(ind))
    X=np.zeros([784,50000])
    for i in range(50000):
        X[:,i]=train_X[ind[i]].flatten()
    y=test_X[random.sample(range(np.shape(test_y)[0]),1)[0]].flatten()
    return(X,y)

def gen_data_PIE():
    """
    Generation of PIE Dataset

    The PIE face image data set used in this experiment contains 11554 gray 
    face images of 68 people, taken under different poses, illumination
    conditions and expressions. The vector y represents one image and the 
    remaining image are in the matrix X.

    Returns
    -------  
    X : np.array of shape (1024, 11553)
        The remaining images.
    y : np.array of shape (1024)
        The image selected.
    """

    list_files = ["data/PIE/PIE05.mat","data/PIE/PIE07.mat","data/PIE/PIE09.mat","data/PIE/PIE27.mat","data/PIE/PIE29.mat"]
    X = np.zeros((1024,11554))
    rows = 0
    for files in list_files:
        content = sio.loadmat(files)
        x = content['fea']
        X[:,rows:(rows+x.shape[0])] = x.T
        rows += x.shape[0]
    indy = random.sample(range(11554),1)
    y = X[:,indy].reshape(-1)
    X = np.delete(X, indy, 1)
    indx = random.sample(range(11553),11553)
    X = X[:,indx]

    return (X,y)