import numpy as np
import matplotlib.pyplot as plt

def landmark(x, y):
    #  = landmark(x_train, x_train)
    return np.exp(-0.1*np.square(x.reshape((-1,1)) - y.reshape((1,-1))))

x_train = np.genfromtxt('../data/ridgetrain.txt', delimiter=' ', usecols=0)
x_test = np.genfromtxt('../data/ridgetest.txt', delimiter=' ', usecols=0)
y_train = np.genfromtxt('../data/ridgetrain.txt', delimiter='  ', usecols=1)
y_test = np.genfromtxt('../data/ridgetest.txt', delimiter='  ', usecols=1)

iter = [2, 5, 20, 50, 100]

for L in iter:
    z = np.random.choice(x_train, L, replace=False)
    Id = np.eye(L)
    xf_train = landmark(x_train, z)

    W = np.dot(np.linalg.inv(np.dot(xf_train.T, xf_train) + 0.1*Id), np.dot(xf_train.T, y_train.reshape((-1,1))))

    xf_test = landmark(x_test, z)

    y_pred = np.dot(xf_test, W)

    rmse = np.sqrt(np.mean(np.square(y_test.reshape((-1,1)) - y_pred)))
    print ('RMSE for lambda = ' + str(L) + ' is ' + str(rmse))

    plt.figure(L)
    plt.title('L = ' + str(L) + ', rmse = ' + str(rmse))
    plt.plot(x_test, y_pred, 'r*')
    plt.plot(x_test, y_test, 'b*')

plt.show()
