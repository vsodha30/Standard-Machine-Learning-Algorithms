import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')


def persReadCSV(path):
    data = []
    kl = 1
    with open(path, 'r') as f:
        for line in f.readlines():
            # line.strip()
            d = line.split(",")
            i, mean = 0, 0
            temp = []
            while (i < len(d)):
                if (d[i] == "NaN" or d[i] == "NaN\n"):
                    mean += 0;
                else:
                    mean += float(d[i]);
    
                i += 1;
            mean = mean/len(d)
#             print('           sdfsdf  ');
#             print(kl)
#             print(mean)
            kl+=1
        
            i=0;
            while (i < len(d)):
                if (d[i] == "NaN" or d[i] == "NaN\n"):
                    temp.append(mean);
                else:
                    temp.append(float(d[i]));
                i += 1;
            data.append(temp);
    return data

def nopersReadCSV():
    data = pd.read_csv('CSE575-HW03-Data.csv',header=0,usecols=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13'])
    Dict={}
    Features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']
    l = []
    for index,rows in data.iterrows():
        list1 = []
        for i in range(len(Features)):
            list1.append(rows[str(Features[i])])
            if Dict.get(str(Features[i])) is None:
                Dict[str(Features[i])] = [rows[str(Features[i])]]
            else:
                Dict[Features[i]].append(rows[Features[i]])
        l.append(list1)
    
    return np.array(l)

# -----------------------------------------------------------------------------
# df = persReadCSV("CSE575-HW03-Data.csv");
ds = nopersReadCSV();
# -----------------------------------------------------------------------------
# myfinalcentroids = None


class Kmeans:
    
    def __init__(self, k, seed = None, max_iter = 200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter
        
            
    
    def initialise_centroids(self, data):
        
        initial_centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_centroids]

        return self.centroids
    
    
    def assign_clusters(self, data):
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        dist_to_centroid =  pairwise_distances(data, self.centroids, metric = 'euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)
        
        return  self.cluster_labels
    
    
    def update_centroids(self, data):
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.k)])
        
        return self.centroids
    
    
    
    def predict(self, data):
        return self.assign_clusters(data)
    
    
    def fit_kmeans(self, data):
        self.centroids = self.initialise_centroids(data)
        
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)          
#             if iter % 100 == 0:
#                 print("Running Model Iteration %d " %iter)
#         print("Model finished running")
        return self   

x = Kmeans(3)
# print(ds.shape)

data = x.initialise_centroids(ds)
# print(data)
data = x.assign_clusters(ds)
# print(data)
data = x.update_centroids(ds)
# print(data)
data = x.predict(ds)
# print(data)
data = x.fit_kmeans(ds)

number_clusters = range(2, 10)

kmeans2 = [Kmeans(k=i, max_iter = 600) for i in number_clusters]
sse = []
for i in range(0, 8):
    kmeans2[i].fit_kmeans(ds)
    
    dist_to_centroid =  pairwise_distances(ds, kmeans2[i].centroids, metric = 'euclidean')
    
    abc = np.argmin(dist_to_centroid, axis = 1)
    ans = 0
    
    for p in range(0,128):
        ans += dist_to_centroid[p][abc[p]]*dist_to_centroid[p][abc[p]]
#         print(ans)
    sse.append(ans)
    
    
plt.plot(number_clusters, sse)
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Objective Function')
plt.title('Objective Function Graph')
plt.show()

x2 = Kmeans(2, max_iter = 1000)
# dt = ds[: , 0 : 2]
x2.fit_kmeans(ds)
# print(x2.centroids)
# print(x2.cluster_labels)

Y_sklearn = ds

fitted = x2.fit_kmeans(Y_sklearn)
predicted_values = x2.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='jet')

centers = fitted.centroids
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);