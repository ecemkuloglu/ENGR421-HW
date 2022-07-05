import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#importing data and divide the data set into two parts

data_set_images = np.genfromtxt("hw03_data_set_images.csv", delimiter=",")
data_set_labels = np.genfromtxt("hw03_data_set_labels.csv")

#Divide the data set into two parts

training_images = np.zeros((25*5, 320))
test_images = np.zeros((14*5,320))
training_labels = np.zeros(25*5).astype(int)
test_labels = np.zeros(14*5).astype(int)

i = 0
while (i < 5):
    training_images[i*25: ((i*25)+25),: ] = data_set_images[i*39: (25+(i*39)),: ]
    test_images[i*14: ((i*14)+14),:] = data_set_images[((i*39)+25): (39+(i*39)),: ]
    
    training_labels[i*25: ((i*25)+25)] = data_set_labels[i*39: (25+(i*39))]
    test_labels[i*14: ((i*14)+14)] = data_set_labels[((i*39)+25): (39+(i*39))]
    i += 1

size = np.array([25,25,25,25,25])


def sigmoid(x, W, v):
    return 1/(1 + np.exp(-(np.matmul(x, W) + v)))

def gradient_W(X, Y_truth, Y_predicted):
    return(np.asarray([-np.matmul(Y_truth[:,c] - Y_predicted[:,c], X) for c in range(5)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))

y = np.stack([1*(training_labels==(i+1)) for i in range(np.max(training_labels))], axis = 1)

W = np.random.uniform(low = -0.01, high = 0.01, size = (training_images.shape[1], 5))
v = np.random.uniform(low = -0.01, high = 0.01, size = (1, 5))

eta = 0.001
epsilon = 0.001
iteration = 1
objective_values = []

while True:

    W_old = W
    v_old = v

    y_predicted = sigmoid(training_images, W, v)
    objective_values = np.append(objective_values, (0.5)*np.sum(np.sum((y - y_predicted)**2, axis= 1), axis = 0))
    
    W = W - eta * gradient_W(training_images, y, y_predicted)
    v = v - eta * gradient_w0(y, y_predicted)
    
    if np.sqrt(np.sum((v - v_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1
    
print(W)
print(v)

#Draw the objective function values
plt.figure(figsize = (12, 10))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

#Calculate the confusion matrix for training set
y_predicted = np.argmax(sigmoid(training_images,W, v), axis = 1) +1
confusion_matrix_training = pd.crosstab(y_predicted, training_labels, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix_training, "\n")


#Calculate the confusion matrix for test set
y_test_predicted =  np.argmax(sigmoid(test_images, W, v), axis = 1) +1
confusion_matrix_test = pd.crosstab(y_test_predicted, test_labels, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix_test)