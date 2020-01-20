import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal as mvn
data = pd.read_csv('CSE575-HW03-Data.csv',header=0,usecols=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13'])

Dict={}

Features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']

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


def initialize_parameters(data):
    l = []
    for index,rows in data.iterrows():
        listnum1 = []
        for i in range(len(Features)):
            listnum1.append(rows[str(Features[i])])
            if Dict.get(str(Features[i])) is None:
                Dict[str(Features[i])] = [rows[str(Features[i])]]
            else:
                Dict[Features[i]].append(rows[Features[i]])
        l.append(listnum1)
    x = np.array(l)
    collistnum1,collist2=[],[]
    labels = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1
, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1
, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1
, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    
    for i in range(2):
        if i==0:
            collistnum1 = x[labels==0]
        else:
            collist2 = x[labels==1]
            
    #print(collistnum1)
    x1 = np.array(collistnum1)
    x2 = np.array(collist2)
    initial_means = np.zeros((2,13))
    initial_cov = np.zeros((2,13,13))
    initial_pi = np.zeros((2))
    for i in range(2):
        initial_pi[i] = 0.5
        if i ==0:
            initial_means[i,:] = np.mean(x1,axis=0)
            #initial_cov[i,:,:] = np.cov(x1.T)
            de_meaned = x1 - initial_means[i,:]
            initial_cov[i,:, :] = np.dot(initial_pi[i] * de_meaned.T, de_meaned) / 64
        else:
            initial_means[i,:] = np.mean(x2,axis=0)
            #initial_cov[i,:,:] = np.cov(x2.T)
            de_meaned = x2 - initial_means[i,:]
            initial_cov[i,:, :] = np.dot(initial_pi[i] * de_meaned.T, de_meaned) / 64
            
    return initial_means,initial_cov,initial_pi,x

        
        

def initparams_X(data,X):
    
    x = X
#     print(x.shape)
    l= x.tolist()
    
    collistnum1,collist2=[],[]
    
    labels = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0
, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0
, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0
, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1])
    for i in range(2):
        if i==0:
            collistnum1 = x[labels==0]
        else:
            collist2 = x[labels==1]
            
    x1 = np.array(collistnum1)
    x2 = np.array(collist2)
    initial_means = np.zeros((2,X.shape[1]))
    initial_cov = np.zeros((2,X.shape[1],X.shape[1]))
    initial_pi = np.zeros((2))
    for i in range(2):
        initial_pi[i] = 0.5
        if i ==0:
            initial_means[i,:] = np.mean(x1,axis=0)
            #initial_cov[i,:,:] = np.cov(x1.T)
            de_meaned = x1 - initial_means[i,:]
            initial_cov[i,:, :] = np.dot(initial_pi[i] * de_meaned.T, de_meaned) / 64
        else:
            initial_means[i,:] = np.mean(x2,axis=0)
            
            de_meaned = x2 - initial_means[i,:]
            initial_cov[i,:, :] = np.dot(initial_pi[i] * de_meaned.T, de_meaned) / 64
            
    return initial_means,initial_cov,initial_pi,x

def _e_step(X, pi, mu, sigma):
    N = X.shape[0]
    
    gamma = np.zeros((N,2))
    for c in range(2):
        gamma[:,c] = pi[c] * mvn.pdf(X, mu[c,:], sigma[c])
    gamma_norm = np.sum(gamma, axis=1)[:,np.newaxis]
    
    gamma /= gamma_norm  
    return gamma


def _m_step( X, gamma,sigma):
    
        N = X.shape[0] # number of objects
        C = gamma.shape[1] # number of clusters
        
        d = X.shape[1] # dimension of each object

        # responsibilities for each gaussian
        pi = np.mean(gamma, axis = 0)

        mu = np.dot(gamma.T, X) / np.sum(gamma, axis = 0)[:,np.newaxis]

        for c in range(C):
            xx = X - mu[c, :] # (N x d)
            
            gamma_diag = np.diag(gamma[:,c])
            x_mu = np.matrix(xx)
            gamma_diag = np.matrix(gamma_diag)
            
            sigma_c = xx.T * gamma_diag * xx
            sigma[c,:,:]=(sigma_c) / np.sum(gamma, axis = 0)[:,np.newaxis][c]
        return pi, mu, sigma
    
    
def _compute_loss_function( X, pi, mu, sigma,gamma):
    
        N = X.shape[0]
        C = gamma.shape[1]
        loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(mu[c], sigma[c],allow_singular=True)
            loss[:,c] = gamma[:,c] * (np.log(pi[c]+0.00001)+dist.logpdf(X)-np.log(gamma[:,c]+0.000001))
        loss = np.sum(loss)
        return loss


def _compute_loss_function2( X, pi, mu, sigma,gamma):
    
        N = X.shape[0]
        C = gamma.shape[1]
        loss = np.zeros((N, C))
        mainsum,sum = 0,0
#         for l in range(128):
#             for j in range(2):
#                 sum = sum + mvn.pdf(X[l,:], mu[j,:], sigma[j])*pi[j]
#             mainsum = mainsum + -np.log(sum)
#             sum=0
#         return mainsum

        for c in range(C):
            dist = mvn(mu[c], sigma[c],allow_singular=True)
            loss[:,c] = gamma[:,c] * (np.log(pi[c]+0.00001)+dist.logpdf(X)-np.log(gamma[:,c]+0.000001))
        loss = np.sum(loss)
        return loss
    
    
def predict( X,pi,mu,sigma):
    
        labels = np.zeros((X.shape[0], 2))
        sum = pi[0]* mvn.pdf(X, mu[0,:], sigma[0])+pi[1]* mvn.pdf(X, mu[1,:], sigma[1])
        for c in range(2):
            labels [:,c] = pi[c] * mvn.pdf(X, mu[c,:], sigma[c])
        labels = labels/sum.reshape(-1,1)
        labels  = labels.argmax(1)
        return labels 
    
def predict_proba( X,mu,sigma):
    
    post_proba = np.zeros((X.shape[0], C))

    for c in range(2):

        post_proba[:,c] = pi[c] * mvn.pdf(X, mu[c,:], sigma[c])

    return post_proba

def fit(data,X=None):
        
        if X is None:
            mu, sigma, pi,X = initialize_parameters(data)#initial_means,initial_cov,initial_pi #
        else:
            mu, sigma,pi,X = initparams_X(data,X)
        
        d = X.shape[1]
        #print(sigma)
        try:
            for run in range(100):  
                gamma  = _e_step(X,pi,mu,sigma)
                #print(gamma)
                pi, mu, sigma = _m_step(X,gamma,sigma)
                loss = _compute_loss_function2(X,pi,mu,sigma,gamma)
                
#                 if run % 10 == 0:
#                     print("Iteration: %d Loss: %0.6f" %(run, loss))

        
        except Exception as e:
            print(e)
            
        
        return mu,sigma,pi,X
    
    
mu,sigma,pi,X = fit(data)

predicted_values = predict(X,pi,mu,sigma)

# print(predicted_values.shape)



centers = np.zeros((2,13))
for i in range(2):
    density = mvn(cov=sigma[i], mean=mu[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]

    
# print(X[:, 1:2].shape)

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):

    ax = ax or plt.gca()
    
    
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        
        width, height=2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
        
plt.figure(figsize = (10,8))
plt.scatter(X[:, 0], X[:, 1],c=predicted_values ,s=50, cmap='jet', zorder=1)    
    
w_factor = 0.2 / pi.max()
print(w_factor)


from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):

    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (13, 13):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        y = 2 * np.sqrt(s)
        width, height = [y[0],y[1]]
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
        
