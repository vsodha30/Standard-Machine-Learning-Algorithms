import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import pairwise_distances
data = pd.read_csv('CSE575-HW03-Data.csv',header=0,usecols=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13'])

Features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']
Dict={}
np.random.seed(100)

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
    
maxf,minf = [],[]
for i in range(len(Features)):
    unique = data[Features[i]].unique()
    maxf.append(np.amax(unique,axis=0))
    minf.append(np.amin(unique,axis=0))
    
bounds = np.array([minf,maxf])
x0 = np.random.uniform(bounds.T[:, 0], bounds.T[:, 1], size=(2, 13))

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
dlmdata = nopersReadCSV();

def initialise_centroids(data,k,df):
        
        Features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']
        maxf,minf = [],[]
        for l in range(len(Features)):
            unique = df[Features[l]].unique()
            maxf.append(np.amax(unique,axis=0))
            minf.append(np.amin(unique,axis=0))
        bounds = np.array([minf,maxf])
        initial_centroids = np.random.uniform(bounds.T[:, 0], bounds.T[:, 1], size=(k, 13))

        
        return initial_centroids,bounds.T
    
def assign_clusters(data,centroids):
    
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        dist_to_centroid =  pairwise_distances(data,centroids, metric = 'euclidean')
        cluster_labels = np.argmin(dist_to_centroid, axis = 1)#Array of indices into the array with axis dim removed
        
        return  cluster_labels
    
    
def update_centroids(data,k,cluster_labels,centroids,bounds):
        for j in range(k):
            jthclusterpts = data[cluster_labels == j]
            

            sum = None
            defnorm = None
            for i in range(jthclusterpts.shape[0]):
                
                fnorm = np.linalg.norm(centroids[j]-jthclusterpts[i])

                if sum is None:
                    sum =  jthclusterpts[i]/fnorm
                else:
                    sum = sum + jthclusterpts[i]/fnorm
                if defnorm is None:
                    defnorm =  1/fnorm
                else:
                    defnorm = defnorm + 1/fnorm
            if sum is not None and defnorm is not None:
                centroids[j] = sum/defnorm
            else:
                centroids[j] = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, 13))
        
        return centroids
    

def predict( data,centroids):        
        return assign_clusters(data,centroids)
    
def Calculate_Squaredsum(data, k, centroids,cluster_labels):
    square_sum=0
    for i in range(k):
        cluster_data = data[cluster_labels == i]
        if cluster_data.size == 0:
            square_sum = square_sum
        else:
            distances = pairwise_distances(cluster_data, [centroids[i]], metric = 'euclidean')
            square_sum += np.sum(distances)
    
    return square_sum


def fit_altkmeans( data,k,df):
        centroids,bounds = initialise_centroids(data,k,df)
        threshold = 0.1
        for iter in range(200):

            cluster_labels = assign_clusters(data,centroids)
            centroids = update_centroids(data,k,cluster_labels,centroids,bounds)
            loss = Calculate_Squaredsum(data, k, centroids,cluster_labels)
            for i in range(data.shape[0]):
                if np.array_equal(centroids[0],data[i]) or np.array_equal(centroids[1],data[i]):
                    loss = Calculate_Squaredsum(data, k, centroids,cluster_labels)
                    return centroids,loss
        loss = Calculate_Squaredsum(data, k, centroids,cluster_labels)
        return centroids,loss
    
centroids,loss = fit_altkmeans(np.array(l),2,data)
predicted_values = predict(np.array(l),centroids)

plt.scatter(np.array(l)[:, 0], np.array(l)[:, 1], c=predicted_values, s=50, cmap='jet')
plt.scatter(centroids[:, 0], centroids[:, 1],c='black', s=300, alpha=0.4);
plt.show()


number_clusters = range(2, 7)
plt.clf()


lossK=[]
for k in number_clusters:
    centroids,loss = fit_altkmeans(np.array(l),k,data)
    
    lossK.append(loss)
plt.plot(number_clusters, lossK)
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Objective Function')
plt.title('Objective Function graph')
plt.show()
