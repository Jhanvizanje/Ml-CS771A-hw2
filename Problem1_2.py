import numpy as np
import math
import matplotlib.pyplot as plt
import random

file_path = "ridgetrain.txt"
train_list = []

with open(file_path, "r") as file:
    for line in file:
        # Split each line into individual values based on whitespace
        values = line.split()
        # Convert the values to floating-point numbers
        float_values = [float(value) for value in values]
        train_list.append(float_values)

# print(train_list)

file_path = "ridgetest.txt"
test_list = []

with open(file_path, "r") as file:
    for line in file:
        # Split each line into individual values based on whitespace
        values = line.split()
        # Convert the values to floating-point numbers
        float_values = [float(value) for value in values]
        test_list.append(float_values)

# print(test_list)
def kernelFunction(x_m,x_n):
    gamma=0.1
    return math.exp(-gamma * (np.linalg.norm(x_m-x_n)**2))

lamda=0.1
L=[2,5,20,50,100]
for l in L:
    x_values = [point[0] for point in train_list]
    random_x_values = random.sample(x_values, l)
    list=[]
    X=[]
    X_test=[]
    for trainData in train_list:
        list=[]
        for r in random_x_values:
            dotP = kernelFunction(r,trainData[0])
            list.append(dotP)
        X.append(list)
    I=np.identity(l)
    I=lamda*I
    Y = [point[1] for point in train_list]
    W = np.dot(np.linalg.inv(np.dot(np.transpose(X),X) + I),np.dot(np.transpose(X),Y))
    for testData in test_list:
        list=[]
        for r in random_x_values:
            dotP = kernelFunction(r,testData[0])
            list.append(dotP)
        X_test.append(list)

    y_output_list=[]
    for x in X_test:
        y_output = np.dot(np.transpose(W),x)
        y_output_list.append(y_output)

    test_x, test_y = zip(*test_list)
    train_x, train_y = zip(*train_list)

    # Create a scatter plot for testdata
    plt.scatter(test_x, test_y, label='True Outputs', marker='o', color='blue')

    # Create a scatter plot for traindata
    plt.scatter(test_x, y_output_list, label='Predicted Outputs', marker='x', color='red')

    # Add labels and a legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title( f'Landmark Ridge Regression (L={l}, lambda={lamda})')
    plt.legend()

    # Show the plot
    plt.show()
    actual = np.array(test_y)
    predicted = np.array(y_output_list)

    # Calculate the squared error for each pair of values
    squared_error = (actual - predicted) ** 2

    # Calculate the mean of the squared errors
    mse = squared_error.mean()

    # Calculate the RMSE by taking the square root of the mean squared error
    rmse = np.sqrt(mse)
    print("L=",l,"RMSE=",rmse)
