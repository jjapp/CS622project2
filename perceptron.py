import numpy as np
import math
import test_utility as tu

def iteration(X, Y, w, b):
    """Takes a training set and updates weights"""
    # stack the arrays
    z = np.column_stack((X, Y))
    # get number of weights
    num_w = np.shape(X)[1]
    new_w = np.copy(w)

    for row in z:
        # get the dot product of w and row
        a = np.dot(new_w, row[:-1]) + b
        y = row[-1]

        if a * y <= 0:
            for i in range(num_w):
                new_w[i] = new_w[i] + y * row[i]
                b = b + y
    return new_w, b


def perceptron_train(X, Y):
    # stack the arrays
    z = np.column_stack((X, Y))

    # get number of weights
    num_w = np.shape(X)[1]

    # create an array to hold the weights
    w = np.zeros(num_w)

    # create the bias variable
    b = 0

    new_weights = iteration(X, Y, w, b)

    while not np.array_equal(w, new_weights[0]):
        w = np.copy(new_weights[0])
        b = new_weights[1]
        new_weights = iteration(X, Y, w, b)

    return new_weights


def perceptron_test(X_test, Y_test, w, b):
    correct = 0
    for i in range(np.shape(X_test)[0]):
        predicted = np.dot(w, X_test[i]) + b
        if predicted < 0:
            predicted = -1
        else:
            predicted = 1
        if predicted == Y_test[i]:
            correct = correct + 1
    accuracy = correct/(np.shape(X_test)[0])
    return accuracy




if __name__ == '__main__':
    X, Y = tu.load_data('data_1.txt')
    z = perceptron_train(X, Y)
    acc = perceptron_test(X, Y, z[0], z[1])
    print(acc)

