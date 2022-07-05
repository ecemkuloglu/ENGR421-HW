import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal

X = np.genfromtxt("hw08_data_set.csv", delimiter=",")
means = np.genfromtxt("hw08_initial_centroids.csv", delimiter=',')

plt.figure(figsize=(10,10))
plt.plot(X[:, 0], X[:, 1],".",color="k" )
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

initial_means = np.array([[+5.0, +5.0],
                        [-5.0, +5.0],
                        [-5.0, -5.0],
                        [+5.0, -5.0],
                        [+5.0, +0.0], 
                        [+0.0, +5.0],
                        [-5.0, +0.0],
                        [+0.0, -5.0],
                        [+0.0, +0.0],]) 

initial_covariances = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+0.2, +0.0], [+0.0, +1.2]],
                             [[+1.2, +0.0], [+0.0, +0.2]],
                             [[+0.2, +0.0], [+0.0, +1.2]],
                             [[+1.2, +0.0], [+0.0, +0.2]],
                             [[+1.6, +0.0], [+0.0, +1.6]]])

class_sizes = np.array([100, 100, 100, 100, 100, 100, 100, 100, 200])

N = X.shape[0]
K = means.shape[0]
D = X.shape[1]

def update_clusters(means, x):
    D = spa.distance_matrix(means, x)
    clusters = np.argmin(D, axis = 0)
    return(clusters)

clusters = update_clusters(means, X)
###################################################

covariances = []
empty_mat = [[0.0, 0.0], [0.0, 0.0]]
for i in range(K):
    for j in range(X[clusters == i].shape[0]):
        cov = np.matmul(((X[clusters == i])[j,:] - means[i,:])[:, None], ((X[clusters == i])[j,:] - means[i,:][None, :]))
        empty_mat += cov
    covariances.append(empty_mat / X[clusters == i].shape[0])
    empty_mat = [[0.0, 0.0], [0.0, 0.0]]
    
priors =[]
for k in range(K):
    priors.append(X[clusters == k].shape[0] / N)
priors = np.array(priors)

ite = 0
while ite < 100:
    p = []
    for k in range(K):
        post = multivariate_normal(means[k], covariances[k]).pdf(X)*priors[k]
        p.append(post)
    H = np.vstack([p[k]/np.sum(p, axis = 0) for k in range(K)])
    
    means = (np.vstack([np.matmul(H[k], X)/np.sum(H[k], axis = 0) for k in range(K)]))
    
    covariances = []
    for k in range(K):
        empty_mat = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(N):
            cov = np.matmul((X[i] - means[k])[:, None], (X[i] - means[k])[None, :])*H[k][i]
            empty_mat += cov
        covariances.append(empty_mat / np.sum(H[k], axis = 0))
    
    priors =[]
    for k in range(K):
        priors.append(np.sum(H[k], axis = 0)/N)
    priors = np.array(priors)
    
    ite +=1
print(means)
clusters=np.argmax(H, axis = 0)

##############################################

plt.figure(figsize = (10, 10))
cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
for c in range(K):
    plt.plot(X[clusters == c, 0], X[clusters == c, 1], ".", markersize = 10, color = cluster_colors[c])
    x,y = np.meshgrid(np.linspace(-8,8,401), np.linspace(-8,8,401))
    plt.contour(x, y, multivariate_normal.pdf(np.concatenate((x.flatten()[:,None], y.flatten()[:,None]), axis = 1), means[c], covariances[c]).reshape(401,401), levels = [0.05], colors = cluster_colors[c])
    plt.contour(x, y, multivariate_normal.pdf(np.concatenate((x.flatten()[:,None], y.flatten()[:,None]), axis = 1), initial_means[c], initial_covariances[c]).reshape(401,401), levels = [0.05], colors = "k", linestyles = "dashed")