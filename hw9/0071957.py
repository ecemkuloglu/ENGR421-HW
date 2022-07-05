import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from numpy.linalg import eig 
from numpy.linalg import matrix_power

X = np.genfromtxt("hw09_data_set.csv", delimiter=",")
delta = 2.0

plt.figure(figsize=(10,10))
plt.plot(X[:, 0], X[:, 1],".",color="k" )
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

###############################
B = np.zeros((X.shape[0],X.shape[0]))
B[spa.distance_matrix(X, X) < delta] = 1
for i in range(X.shape[0]):
    B[i,i] = 0

plt.figure(figsize = (10,10))

for i in range(X.shape[0]):
    for j in range(i, X.shape[0]):
        if B[i][j] == 1:
            plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], "-", linewidth = 0.5, color = "#7f7f7f")

plt.plot(X[:, 0], X[:, 1], ".", markersize = 10, color = "black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
###############################
D = np.diag(np.sum(B, axis = 1))
L = np.identity(X.shape[0]) - np.matmul(np.sqrt(matrix_power(D,-1)), np.matmul(B,np.sqrt(matrix_power(D,-1))))
print("L:\n", L[0:5, 0:5])
###############################
eingenvalues , eigenVectors = eig(L)
Z = eigenVectors[:,np.argsort(eingenvalues)[1:(6)]]
print("Z:\n", Z[0:5, 0:5])

###########TAKEN FROM LAB############
K = 9     #cluster number
rows = [242, 528, 570, 590, 648, 667, 774, 891, 955]
centroids = Z[rows,:]

def update_centroids(memberships, X):
    centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

memberships = update_memberships(centroids, Z)
iteration = 1
while True:
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break
    
    iteration = iteration + 1

centroids = update_centroids(memberships, X)
cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

plt.figure(figsize=(10,10))
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10, color = cluster_colors[c])
    plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12, markerfacecolor=cluster_colors[c], markeredgecolor="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()



