from keras.datasets import mnist
import scipy.io as sio
import random
import itertools
import numpy as np

def gen_data_MNIST():
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

    return (X,y)