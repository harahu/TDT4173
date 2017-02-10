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

def E_mse(X, y, w):
    return (1/len(y))*norm(X @ w - y)**2

def h(x, w):
    return w.T @ x

def plot2D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

def plot1D(inS0, outS0, inS1, outS1, inP, outP):
    fig, ax = plt.subplots()
    ax.scatter(inS0, outS0, marker='x', label='Training Data')
    ax.scatter(inS1, outS1, marker='x', label='Test Data')
    ax.plot(inP, outP, c='r', label='Hypothesis')
    legend = ax.legend(loc='upper center', shadow=True)
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

def main():
    regression2D()
    regression1D()

if __name__ == '__main__':
    main()
