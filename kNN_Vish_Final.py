import numpy as np
import pandas as pd
import gzip
import struct
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plotgraph
import matplotlib.cm as cm
import random
from numpy import *
%matplotlib inline

np.random.seed(1)

def read(dataset = "training", path = "./MNIST/"):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (lbl, img)

class simple_knn():
    "a simple kNN with L2 distance"

    def _init_(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            k_closest_y = labels[:k]
            
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return(y_pred)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis = 1)
        sum_square_train = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)

        return(dists)

def read_file(filename):
    with gzip.open(filename) as f:
        dims = struct.unpack('>HBB', f.read(4))[2]
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
    
    
fileData = read_file('data/train-images-idx3-ubyte.gz')
trainingX = np.reshape(fileData, (60000, 28 * 28))
trainingY = read_file('data/train-labels-idx1-ubyte.gz')
fileData = read_file('data/t10k-images-idx3-ubyte.gz')
testingX = np.reshape(fileData, (10000, 28 * 28))
testingY = read_file('data/t10k-labels-idx1-ubyte.gz')

# plotgraph.imshow(trainingX[0,:].reshape(28,28))

distance_array = cdist(testingX, trainingX, 'euclidean')
# print(distance_array.shape)

def nearestNeighbors(trainingX, i, k):
    distances = []
    n = trainingX.shape[0]
    for j in range(len(trainingX)):
        distances.append(distance_array[i][j])
    neighbors_idxs = np.argsort(distances)[:k]
    fullneighbors_idxs = np.argsort(distances)
    return neighbors_idxs,fullneighbors_idxs

def predict(neighbors, trainingY):
    results = []
    for id in neighbors:
        results.append(trainingY[int(id)])
    results = np.array(results)
    return np.argmax(np.bincount(results));

def customKNN(trainingX, trainingY, testingX, k):
    predictions = []
    allNN = np.zeros((testingX.shape[0],trainingX.shape[0]))
    for i in tqdm(range( len(testingX) )):
        neighbors,fullneighbours = nearestNeighbors(trainingX, i, k)
        allNN[i] = fullneighbours
        predictions.append(predict(neighbors, trainingY))

    return np.array(predictions),allNN


def customKNN1(trainingX, trainingY, testingX, k,allNN):
    predictions = []
    for i in tqdm(range( len(testingX) )):
        neighbors = allNN[i,:k]
        predictions.append(predict(neighbors, trainingY))

    return np.array(predictions)

def accuracy(predictions, testingY):
    return np.sum(predictions == testingY) / len(testingY)

k=[1,3,5,10,20,30,40,50,60]
accuracy_of_customknn=[]
allNN = np.zeros((testingX.shape[0],trainingX.shape[0]))
for i in k:
    if i==1:
        predictions,allNN = customKNN(trainingX, trainingY, testingX, i)
        accuracy_of_customknn.append(accuracy(predictions, testingY))
    else:
        predictions = customKNN1(trainingX, trainingY, testingX, i,allNN)
        accuracy_of_customknn.append(accuracy(predictions, testingY))

predictions = customKNN(trainingX, trainingY, testingX, 3)
accuracy1=accuracy(predictions,testingY)

plotgraph.plot(k,accuracy_of_customknn)