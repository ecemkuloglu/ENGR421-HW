
import numpy as np
import pandas as pd 

#importing data

data_set_images = np.genfromtxt("hw02_data_set_images.csv", delimiter=",")
data_set_labels = np.genfromtxt("hw02_data_set_labels.csv")

#Divide the data set into two parts

training_images = np.zeros((25*5, 320))
test_images = np.zeros((14*5,320))
training_labels = np.zeros(25*5).astype(int)
test_labels = np.zeros(14*5).astype(int)

i = 0
while (i < 5):
    training_images[i*25: ((i*25)+25),: ] = data_set_images[i*39: (25+(i*39)),: ]
    training_labels[i*25: ((i*25)+25)] = data_set_labels[i*39: (25+(i*39))]
    test_images[i*14: ((i*14)+14),:] = data_set_images[((i*39)+25): (39+(i*39)),: ]
    test_labels[i*14: ((i*14)+14)] = data_set_labels[((i*39)+25): (39+(i*39))]
    i += 1
    
#Estimate the parameters
sample_means = np.empty((0,320))
for i in range (5):
    sample_means = np.append(sample_means, np.sum(training_images[i*25: ((i*25)+25),: ], axis = 0, keepdims=True)/25 ,axis=0)

print("Sample Means:")
print(sample_means)

class_priors = np.array([25, 25, 25, 25, 25])/125
print("Class Priors:")
print(class_priors)

#score calculation

def score_function(data):
    scores = []
    for point in data:
        score=[]
        for i in range(5):
            score.append(np.dot(point,np.log(sample_means[i] + 1e-100)) + 
                                     np.dot(1-point, np.log(1-sample_means[i] + 1e-100)) + 
                                     np.log(class_priors[i] + 1e-100))
        scores.append(score)      
    score1 = np.array(scores)
    return score1
    

#Calculate the confusion matrix for training set
y_training_predicted = np.argmax(score_function(training_images),axis = 1) +1

confusion_matrix_training = pd.crosstab(y_training_predicted, training_labels, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix_training)

#Calculate the confusion matrix for test set
y_test_predicted =  np.argmax(score_function(test_images),axis = 1) +1

confusion_matrix_test = pd.crosstab(y_test_predicted, test_labels, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix_test)