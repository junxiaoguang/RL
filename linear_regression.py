__author__ = 'Jxg'
import numpy as np
import pylab

def optimizer(x, y,starting_b,starting_m,learning_rate,num_iter):
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iter):
        # update b and m with the new more accurate b and m by performing
        b, m = compute_gradient(b, m, x, y, learning_rate)
    return b, m

def compute_gradient(b_current,m_current,x, y ,learning_rate):
    N = float(len(x))

    # Vectorization implementation
    predict = model(b_current, m_current, x)
    b_gradient = (1.0/N)*((predict - y)*x).sum()
    m_gradient = (1.0/N)*(predict - y).sum()

    # update parameter
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m

def model(a, b, x):
    return a*x + b

def plot_data(x, y, b, m):

    #plottting
    y_predict = model(b, m, x)
    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.show()


def Linear_regression():
    # get train data
    data = np.loadtxt('data.csv', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]

    learning_rate = 0.001

    # initial theta
    initial_b = 0.0
    initial_m = 0.0

    # Gradient learning number
    num_iter = 1000

    # train model
    b, m = optimizer(x, y, initial_b, initial_m, learning_rate, num_iter)

    # plot result
    plot_data(x, y, b, m)

if __name__ =='__main__':

    Linear_regression()
