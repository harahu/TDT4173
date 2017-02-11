import math
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_dataset(filename):
    """Collects input and output vectors from csv file.
    """
    with open(filename, 'r') as f:
        dataset = [line.rstrip('\n') for line in f]
    X = [[float(num) for num in line.split(',')] for line in dataset]
    y = []
    for vector in X:
        vector.insert(0, 1.0)
        y.append(vector.pop())
    return np.array(X), np.array(y)

def ols(X, y):
    return pinv(X.T @ X) @ X.T @ y

def sigmoid(z):
    return (1.0/(1.0+math.exp(-z)))

def logistic_gd(X, y):
    eta = 0.05
    w = np.array([0 for _ in range(3)])
    for _ in range(3000):
        w = w - eta*dE_ce(X, y, w)
    return w

def E_mse(X, y, w):
    return (1/len(y))*norm(X @ w - y)**2

def E_ce(X, y, w):
    #TODO
    pass

def dE_ce(X, y, w):
    dE = np.zeros(3)
    for xi, yi in zip(X, y):
        dE += (sigmoid(h(xi, w)) - yi) * xi
    return dE

def h(x, w):
    return w.T @ x

def classify_z(z):
    return 1 if z > 0 else 0

def plot2D():
    #TODO
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

def plot1D(inS0, outS0, inS1, outS1, inP, outP):
    fig, ax = plt.subplots()
    ax.scatter(inS0, outS0, marker='x', label='Training Data')
    ax.scatter(inS1, outS1, marker='x', label='Test Data')
    ax.plot(inP, outP, c='r', label='Hypothesis')
    ax.legend(loc='upper center', shadow=True)
    plt.savefig("1d-linreg.png")
    plt.show()

def d_line(x, w):
    return -(w[0]+w[1]*x)/w[2]

def split(X, y):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for xi, yi in zip(X, y):
        if yi == 0:
            x0.append(xi[1])
            y0.append(xi[2])
        else:
            x1.append(xi[1])
            y1.append(xi[2])
    return x0, y0, x1, y1

def plot_class(X, y, Xtest, ytest, w):
    x0, y0, x1, y1 = split(X, y)
    x0t, y0t, x1t, y1t = split(Xtest, ytest)
    d_line_x = [0, 1]
    d_line_y = [d_line(x, w) for x in d_line_x]
    fig, ax = plt.subplots()
    ax.set_ylim([-0.1,1.1])
    ax.scatter(x1, y1, marker='o', c='g', label='Class 1 Train')
    ax.scatter(x0, y0, marker='o', c='r', label='Class 0 Train')
    ax.scatter(x1t, y1t, marker='o', c='#005000', label='Class 1 Test')
    ax.scatter(x0t, y0t, marker='o', c='#500000', label='Class 0 Test')
    ax.plot(d_line_x, d_line_y, 'b--')
    ax.legend(loc='upper center', shadow=True)
    plt.show()


def regression2D():
    X, y = get_dataset('datasets/regression/reg-2d-train.csv')
    Xtest, ytest = get_dataset('datasets/regression/reg-2d-test.csv')
    w = ols(X, y)
    print(w)
    print(E_mse(X, y, w))
    print(E_mse(Xtest, ytest, w))

def regression1D():
    X, y = get_dataset('datasets/regression/reg-1d-train.csv')
    Xtest, ytest = get_dataset('datasets/regression/reg-1d-test.csv')
    w = ols(X, y)
    print(E_mse(X, y, w))
    print(E_mse(Xtest, ytest, w))
    inS = [x[1] for x in X]
    inP = np.array(sorted(inS))
    inS1 = [x[1] for x in Xtest]
    outP = [h(np.array([1, x]), w) for x in inP]
    plot1D(inS, y, inS1, ytest, inP, outP)

def classification():
    X, y = get_dataset('datasets/classification/cl-train-1.csv')
    Xtest, ytest = get_dataset('datasets/classification/cl-test-1.csv')
    w = logistic_gd(X, y)
    zs = X @ w
    pred = np.array([classify_z(z) for z in zs])
    plot_class(X, y, Xtest, ytest, w)

def main():
    #regression2D()
    #regression1D()
    classification()

if __name__ == '__main__':
    main()
