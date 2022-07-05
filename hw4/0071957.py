import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#2 export data
train_data = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
test_data = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")

x_train = train_data[1:,0]
y_train = train_data[1:,1]

x_test = test_data[1:,0]
y_test = test_data[1:,1]

min = np.around(min(x_train))
max = np.around(max(x_train))
data_interval = np.linspace(min, max, 2001)

K = np.max(y_train)
N = train_data.shape[0]

#regressogram
h = 0.1
left_borders = np.arange(min, max, h)
right_borders = np.arange(min + h, max + h, h)

y_test_p = np.asarray([np.average(y_train[((left_borders[b] < x_train) & (x_train <= right_borders[b]))]) for b in range(len(left_borders))]) 


#training
plt.figure(figsize=(10,6))
plt.plot(x_train[:], y_train[:], "b.", label = "training")
plt.legend()

plt.plot(data_interval[1:], np.repeat(y_test_p,100), "k")
plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#test
plt.figure(figsize=(10,6))
plt.plot(x_test[:], y_test[:], "r.", label = "test")
plt.legend()

plt.plot(data_interval[1:], np.repeat(y_test_p,100), "k")   
plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

y_test_p2 = np.zeros((x_test.shape[0],))
#4 regressogram (RMSE)
for i in range(x_test.shape[0]):
    for b in range(len(left_borders)):
        if(((left_borders[b] < x_test[i]) & (x_test[i] <= right_borders[b]))):
            y_test_p2[i]= y_test_p[b] 
            

RMSE = np.sqrt(np.sum((y_test - y_test_p2)**2)/ y_test.shape[0])
print("Regressogram => RMSE is", RMSE, "when h is", h)

#5 running mean smoother
h = 0.1
left_borders = np.arange(min, max, h)
right_borders = np.arange(min + h, max + h, h)
y_test_pred = np.asarray([np.average(y_train[((i- 0.5 * h) < x_train) & (x_train <= (i + 0.5 * h))]) for i in data_interval])

#training
plt.figure(figsize=(10,6))
plt.plot(x_train[:], y_train[:], "b.", label = "training")
plt.legend()

plt.plot(data_interval, y_test_pred, "k-")

plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#test
plt.figure(figsize=(10,6))
plt.plot(x_test[:], y_test[:], "r.", label = "test")
plt.legend()

plt.plot(data_interval, y_test_pred, "k-")

plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#6 RMSE of running mean smoother
y_test_pred2 = np.zeros((x_test.shape[0],))
y_test_pred2 = np.asarray([np.average(y_train[((i- 0.5 * h) < x_train) & (x_train <= (i + 0.5 * h))]) for i in x_test])

RMSE = np.sqrt(np.sum((y_test - y_test_pred2)**2)/ y_test.shape[0])
print("Running Mean Smoother => RMSE is", RMSE,  "when h is", h)

#7 Kernel smoother
h = 0.02

y_test_prediction = np.asarray([(np.sum((y_train) * (1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (j - x_train)**2 / h**2)))) 
                                 / np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (j - x_train)**2 / h**2)) for j in data_interval])


plt.figure(figsize=(10,6))
plt.plot(x_train[:], y_train[:], "b.", label = "training")
plt.legend()

plt.plot(data_interval, y_test_prediction, "k") 

plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#test
plt.figure(figsize=(10,6))
plt.plot(x_test[:], y_test[:], "r.", label = "test")
plt.legend()


plt.plot(data_interval, y_test_prediction, "k") 

plt.xlabel("Time (sec)")
plt.ylabel("Signal (milivolt)")
plt.show()

#8 RMSE of your kernel smoother
y_test_prediction2 = np.zeros((x_test.shape[0],))
y_test_prediction2 = np.asarray([(np.sum((y_train) * (1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / h**2)))) 
                                 / np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / h**2)) for x in x_test])

RMSE = np.sqrt(np.sum((y_test - y_test_prediction2)**2)/ y_test.shape[0])
print("Kernel Smoother => RMSE is", RMSE, "when h is", h)


    