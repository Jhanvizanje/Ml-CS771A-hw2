import numpy as np
import math
import matplotlib.pyplot as plt
file_path = "ridgetrain.txt"
train_list = []

with open(file_path, "r") as file:
    for line in file:
        # Split each line into individual values based on whitespace
        values = line.split()
        # Convert the values to floating-point numbers
        float_values = [float(value) for value in values]
        train_list.append(float_values)

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

n = len(train_list)
kernel_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        kernel_matrix[i][j] = kernelFunction(train_list[i][0],train_list[j][0])

lamda = [0.1,1,10,100]
yVector = [x[1] for x in train_list]
alpha = []
for L in lamda:
    I=np.identity(n)
    I=L*I
    kernel_matrix_temp=kernel_matrix+I
    kernel_matrix_temp=np.linalg.inv(kernel_matrix_temp)
    alpha = np.dot(kernel_matrix_temp,yVector)
    y_output_list=[]
    for x_test in test_list:
        result_vector = []
        for x_train in train_list:
            dotProduct = kernelFunction(x_test[0],x_train[0])
            result_vector.append(dotProduct)
        y_output = np.dot(result_vector, np.transpose(alpha))
        y_output_list.append(y_output)

    test_x, test_y = zip(*test_list)
    train_x, train_y = zip(*train_list)

    # Create a scatter plot for testdata
    plt.scatter(test_x, test_y, label='True Output', color='blue')

    # Create a scatter plot for traindata
    plt.scatter(test_x, y_output_list, label='Predicted Output', color='red')

    # Add labels and a legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('lambda='+str(L))
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
    print(f'RMSE for Kernel Ridge Regression (lambda={L}): {rmse}')

