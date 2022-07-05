#71957

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.colors as colors
import scipy.linalg as linalg

#random data points parameters
np.random.seed(17)

M1 = np.array([0.0, +4.5])
M2 = np.array([-4.5, -1.0])
M3 = np.array([+4.5, -1.0])
M4 = np.array([0.0, -4.0])


cov1 = np.array([[+3.2, 0.0], [0.0, +1.2]])
cov2 = np.array([[+1.2, 0.8], [0.8, +1.2]])
cov3 = np.array([[+1.2, -0.8], [-0.8, +1.2]])
cov4 = np.array([[+1.2, 0.0], [0.0, +3.2]])

N1 = 105
N2 = 145
N3 = 135
N4 = 115

#data generation

points1 = np.random.multivariate_normal(M1, cov1, N1)
points2 = np.random.multivariate_normal(M2, cov2, N2)
points3 = np.random.multivariate_normal(M3, cov3, N3)
points4 = np.random.multivariate_normal(M4, cov4, N4)

points = np.concatenate([points1, points2, points3, points4])

y = np.concatenate((np.repeat(1, N1), 
                   np.repeat(2, N2), 
                   np.repeat(3, N3), 
                   np.repeat(4, N4)))


#estimate parameters 

sample_means = np.stack([np.mean(points[y == (c+ 1)], axis=0) for c in range(4)])

sample_covariances = np.stack([np.cov(np.transpose(points[y==(c+1)]-sample_means[c])) for c in range(4)])

class_priors = [np.mean(y == c+1) for c in range(4)]

#Calculate the confusion matrix

noneclass = np.empty((0,2), float)
confusion_matrix = np.zeros((4,4), dtype=int)
c1=  -(1/2)*np.log(2*math.pi)
eye2 = np.eye(2)
for i in points1:
    data = np.argmax(np.stack([c1*np.log(np.linalg.det(sample_covariances[c]))
            -(1/2)*np.matmul((i-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), 
            eye2), np.transpose((i- sample_means[c]))))+np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (data != 0):
        confusion_matrix[data,0] += 1
        noneclass =np.append(noneclass, i.reshape(1,2), axis = 0)
    else:
        confusion_matrix[0,0] +=1 
        
for i in points2:
    data = np.argmax(np.stack([c1*np.log(np.linalg.det(sample_covariances[c]))
            -(1/2)*np.matmul((i-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), 
            eye2), np.transpose((i- sample_means[c]))))+np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (data != 1):
        confusion_matrix[data,1] += 1
        noneclass =np.append(noneclass, i.reshape(1,2), axis = 0)
    else:
        confusion_matrix[1,1] +=1 

for i in points3:
    data = np.argmax(np.stack([c1*np.log(np.linalg.det(sample_covariances[c]))
            -(1/2)*np.matmul((i-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), 
            eye2), np.transpose((i- sample_means[c]))))+np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (data != 2):
        confusion_matrix[data,2] += 1
        noneclass =np.append(noneclass, i.reshape(1,2), axis = 0)
    else:
        confusion_matrix[2,2] +=1 
        
for i in points4:
    data = np.argmax(np.stack([c1*np.log(np.linalg.det(sample_covariances[c]))
            -(1/2)*np.matmul((i-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), 
            eye2), np.transpose((i- sample_means[c]))))+np.log(class_priors[c]) for c in range(4)]), axis=0)
    if (data != 3):
        confusion_matrix[data,3] += 1
        noneclass =np.append(noneclass, i.reshape(1,2), axis = 0)
    else:
        confusion_matrix[3,3] +=1 

print("confusion matrix:")
print("y_truth 1  2  3  4")
print("y_pred ")
for i in range(4):
    print(i+1,end = "     ")
    for j in range(4):
        print(confusion_matrix[i,j], end="  ")
    print("\n")
        
#score

points , y = np.meshgrid(np.linspace(-10,10,150), np.linspace(-10,10,150))
vector = np.stack((points.reshape(150**2), y.reshape(150**2)), axis =1)
scores = np.array([[]]*4)

for k in vector:
    score = np.stack([c1*np.log(np.linalg.det(sample_covariances[c]))
            -(1/2)*np.matmul((k-sample_means[c]), np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), 
            eye2), np.transpose((k- sample_means[c]))))+np.log(class_priors[c]) for c in range(4)])
    
    scores = np.append(scores, score.reshape((4,1)), axis=1)
    datas = np.argmax(scores, axis=0)


#Draw decision boundaries
plt.figure(figsize = (16,16))

plt.plot(points1[:,0], points1[:,1], "r.", markersize = 16)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 16)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 16)
plt.plot(points4[:,0], points4[:,1], "m.", markersize = 16)
plt.plot(noneclass[:,0], noneclass[:,1], "ok", markersize = 18, markerfacecolor ="none")

plt.pcolormesh(points, y, datas.reshape(150,150), cmap=colors.ListedColormap(["red", "green", "blue", "magenta" ]), alpha=0.5)

plt.xlabel("$x$")
plt.xlabel("$y$")
plt.show()