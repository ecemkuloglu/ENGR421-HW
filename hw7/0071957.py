
import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
import scipy.stats as stats
import scipy.linalg as linalg



# read data into memory
data_set = np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
label_set = np.genfromtxt("hw07_data_set_labels.csv", delimiter = ",")

x = data_set[:2000]
y = label_set[:2000]

x_test = data_set[2000:]
y_test = label_set[2000:]

N = len(y)
D = x.shape[1]
K = int(max(y))

############################################

def mean(x, y, K):
    means = []
    for c in range(K):
        means.append([np.mean(x[y == c+1],axis=0)])
    return means

means = mean(x, y ,K)

within_m = np.zeros((D,D))
between_m = np.zeros((D,D))

def within_matrix(x,y,means,K,within_m):
    within = [(np.dot(np.transpose(x[y == (c + 1)] - means[c]), (x[y == (c + 1)] - means[c]))) for c in range(K)]
    for i in range(K):
        within_m += within[i]
    return within_m

def between_matrix(x,y,means,K,between_m):
    for i in range(K):
        between_m += ((x[y == i+1]).shape[0]) * np.dot((((np.mean(x[y == i+1], axis = 0)) - (np.mean(means, axis = 0))).reshape(D,1)), np.transpose((((np.mean(x[y == i+1], axis = 0)) - (np.mean(means, axis = 0))).reshape(D,1))))
    return between_m
  

within_m =within_matrix(x,y,means,K,within_m)
between_m=  between_matrix(x,y,means,K,between_m)   

print(within_m[0:4,0:4])
print(between_m[0:4,0:4])

values, vectors = np.linalg.eig(np.dot(np.linalg.inv(within_m), between_m))
values = np.real(values)
vectors = np.real(vectors)
print(values[0:9])


############################################
# calculate two-dimensional projections
z_train = np.matmul(x - np.mean(x, axis = 0), vectors[:,[0, 1]])

# plot two-dimensional projections
plt.figure(figsize = (10, 10))
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(z_train[y == c + 1, 0], z_train[y == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.ylim(-6,6)
plt.xlim(-6,6)
plt.show()

############################################
# calculate two-dimensional projections
z_test = np.matmul(x_test - np.mean(x_test, axis = 0), vectors[:,[0, 1]])

# plot two-dimensional projections
plt.figure(figsize = (10, 10))
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(z_test[y_test == c + 1, 0], z_test[y_test == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.ylim(-6,6)
plt.xlim(-6,6)
plt.show()

####################################################

z_train = np.dot(x - np.mean(x, axis = 0), vectors[:,0:9])
z_test = np.dot(x_test - np.mean(x_test, axis = 0), vectors[:,0:9])

def knn(z, y):
    knn=[]
    for i in range(len(z[:,1])):
        distances = []
        for j in range(len(z[:,1])):
            distances.append(dt.euclidean(z[i,:],z[j,:]))
        l=sorted(distances)
        classes= []
        for k in range(11):
            index= distances.index(l[k])
            classes.append(int(y[index]))
        knn.append(stats.mode(classes)[0])
    return np.array(knn)


train_knn = knn(z_train, y)
test_knn = knn(z_test, y_test)

cm_train = pd.crosstab(train_knn[:,0], y, rownames = ['y_hat'], colnames = ['y_train'])
print(cm_train)

cm_test = pd.crosstab(test_knn[:,0], y_test, rownames = ['y_hat'], colnames = ['y_test'])
print(cm_test)










