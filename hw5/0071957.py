import numpy as np
import matplotlib.pyplot as plt

###################### importing data ########################

train_data = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
test_data = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

x_train = train_data[1:,0]
y_train = train_data[1:,1]

x_test = test_data[1:,0]
y_test = test_data[1:,1]

# get number of classes, number of samples, and number of features
K = np.max(y_train)
N = x_train.shape[0]
D = 1

# get train and test splits


############################ part 2 #########################################

def rmse(y, y_predicted):
    RMSE = np.sqrt((np.sum((y - y_predicted)**2)) / y.shape[0])
    return RMSE


def tree(p):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    y_pred = {}

    # get numbers of train and test samples
    train_n = len(y_train)
    test_n = len(y_test)

    # put all training instances into the root node
    node_indices[1] = np.array(range(train_n))
    is_terminal[1] = False
    need_split[1] = True
    
    # create necessary data structures
    while True:
        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
    
            if len(y_train[data_indices]) <= p:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                best_scores = np.repeat(0.0, D)
                best_splits = np.repeat(0.0, D)
                for d in range(D):
                    unique_values = np.sort(np.unique(x_train[data_indices]))
                    split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                    split_scores = np.repeat(0.0, len(split_positions))
                    for s in range(len(split_positions)):
                        left_indices = data_indices[x_train[data_indices] > split_positions[s]]
                        right_indices = data_indices[x_train[data_indices] <= split_positions[s]]
                        split_scores[s] = (1 / len(data_indices) * (np.sum((y_train[left_indices] - np.mean(y_train[left_indices]))** 2))) + (1 / len(data_indices) * np.sum(( y_train[right_indices] - np.mean(y_train[right_indices])) ** 2))
                    best_scores[d] = np.min(split_scores)
                    best_splits[d] = split_positions[np.argmin(split_scores)]
                # decide where to split on which feature
                split_d = np.argmin(best_scores)
                node_features[split_node] = split_d
                node_splits[split_node] = best_splits[split_d]
            
                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices] > best_splits[split_d]]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
                
      
                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] <= best_splits[split_d]]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    for key in node_indices.keys():
        y_pred[key]= np.mean(y_train[node_indices[key]])
        
    y_predicted_train = []
    for i in range(x_train.shape[0]):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted_train.append(y_pred[index])
                break
            else:
                if x_train[i] > node_splits[index]:
                    index = 2*index
                else: 
                    index = 2*index + 1
                    
    y_predicted_train = np.array(y_predicted_train)
    
    rmse_train = rmse(y_train, y_predicted_train)
    
    y_predicted_test = []
    for i in range(x_test.shape[0]):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted_test.append(y_pred[index])
                break
            else:
                if x_test[i] > node_splits[index]:
                    index = 2*index
                else: 
                    index = 2*index + 1
                    
    y_predicted_test = np.array(y_predicted_test)
    
    rmse_test = rmse(y_test, y_predicted_test)
  
    return is_terminal, node_splits, y_pred, rmse_train, rmse_test




######################### part 3#######################################
P = 30
is_terminal, node_splits, y_pred, rmse_train, rmse_test = tree(P)


print("RMSE on training set is", rmse_train ," when P is", P)
print("RMSE on test set is",rmse_test ," when P is", P)

x_interval = np.linspace(0,2,1001)
y_predicted = []
for i in range(x_interval.shape[0]):
    index = 1
    while True:
        if is_terminal[index] == True:
            y_predicted.append(y_pred[index])
            break
        else:
            if x_interval[i] > node_splits[index]:
                index = 2*index
            else: 
                index = 2*index + 1
                
y_predicted = np.array(y_predicted)

#training
plt.figure(figsize=(10,6))
plt.plot(x_train[:], y_train[:], "b.", label = "training")
plt.legend()

plt.plot(x_interval , y_predicted,"k-")

plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#test
plt.figure(figsize=(10,6))
plt.plot(x_test[:], y_test[:], "r.", label = "test")
plt.legend()

plt.plot(x_interval , y_predicted,"k-")

plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

##############################

train_rmse =[]
test_rmse =[]

for p in range (10, 55, 5):
    is_terminal, node_splits, y_pred, rmse_train, rmse_test = tree(p)
    train_rmse.append(rmse_train) 
    test_rmse.append(rmse_test) 



plt.figure(figsize = (12,6))
plt.plot(range(10, 55, 5), train_rmse[:],'.b-', markersize = 10, label = "training")
plt.plot(range(10, 55, 5), test_rmse[:], '.r-', markersize = 10, label = "test")

plt.xlabel("Pre-prunning size -P-")
plt.ylabel("RMSE")
plt.legend()
plt.show()

