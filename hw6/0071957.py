import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
import scipy.stats as stats
import scipy.linalg as linalg

# read data into memory
data_set = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
label_set = np.genfromtxt("hw06_data_set_labels.csv", delimiter = ",")

x= data_set[:,:]
x_train = x[:1000, :]
x_train = np.array(x_train)

x_test = x[1000:2000, :]
x_test = np.array(x_test)

y = label_set.astype(int)
y_train = y[:1000]
y_test = y[1000:2000]

# get number of samples and number of features
N_train = len(y_train)
N_test = len(y_test)
D_train = x_train.shape[1]

K = np.max(y_train)

H_train = []
H_test= []

for i in range(1000):
    images = x_train[i].reshape(-1, 28)
    h,bin = np.histogram(images.ravel(),64)
    H_train.append(np.array(h/784))
H_train = np.array(H_train)

for i in range(1000):
    images = x_test[i].reshape(-1, 28)
    h,bin = np.histogram(images.ravel(),64)
    H_test.append(np.array(h/784))
H_test = np.array(H_test)

print(H_train[0:5, 0:5])
print(H_test[0:5, 0:5])

####################################
# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


#calculate Gaussian kernel
s = 1
K_train = gaussian_kernel(x_train, x_train, s)
yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train

# set learning parameters
C = 10  
epsilon = 0.001

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * y_train[None,:])
b = cvx.matrix(0.0)
                    
# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha = np.reshape(result["x"], N_train)
alpha[alpha < C * epsilon] = 0
alpha[alpha > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha != 0)
active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
################################################################

#################confusion_matrix_train#####################
f_predicted = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0

# calculate confusion matrix
y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix_train = pd.crosstab(np.reshape(y_predicted, N_train), y_train,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix_train)

################confusion_matrix_test#############################
f_predicted = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0

# calculate confusion matrix
y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix_train = pd.crosstab(np.reshape(y_predicted, N_test), y_test,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix_train)



#############################################
plt.figure(figsize = (12 , 8))

plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
