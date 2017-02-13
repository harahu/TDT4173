import math, os
import numpy as np
from numpy.linalg import pinv, norm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath('__file__'))

r1train = os.path.join(script_dir, 'datasets', 'regression', 'reg-1d-train.csv')
r1test = os.path.join(script_dir, 'datasets', 'regression', 'reg-1d-test.csv')
r2train = os.path.join(script_dir, 'datasets', 'regression', 'reg-2d-train.csv')
r2test = os.path.join(script_dir, 'datasets', 'regression', 'reg-2d-test.csv')

c1train = os.path.join(script_dir, 'datasets', 'classification', 'cl-train-1.csv')
c1test = os.path.join(script_dir, 'datasets', 'classification', 'cl-test-1.csv')
c2train = os.path.join(script_dir, 'datasets', 'classification', 'cl-train-2.csv')
c2test = os.path.join(script_dir, 'datasets', 'classification', 'cl-test-2.csv')

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
    """Calculates weight vector through the ordinary least squares method
    """
    return pinv(X.T @ X) @ X.T @ y

def sigmoid(z):
    """Sigmoid/logistic function
    """
    return (1.0/(1.0+math.exp(-z)))

def logistic_gd(X, y, w, eta):
    """Performs one round of batch gradient descent on w
    """
    return w - eta*dE_ce(X, y, w)

def E_mse(X, y, w):
    """Mean squared error function
    """
    e = X @ w - y
    return (1/len(y)) * (e.T @ e)

def E_ce(X, y, w):
    """Cross-entropy error function
    """
    e = 0
    for xi, yi in zip(X, y):
        z = h(xi, w)
        e += yi*math.log(sigmoid(z)) + (1 - yi)*math.log(1 - sigmoid(z))
    e *= -(1/len(y))
    return e

def dE_ce(X, y, w):
    """Partial derivative of the cross-entropy error
    with respect to w
    """
    dE = np.zeros(3)
    for xi, yi in zip(X, y):
        dE += (sigmoid(h(xi, w)) - yi) * xi
    return dE

def h(x, w):
    """Returns h(x), where h is a linear hypothesis based on weights w
    """
    return w.T @ x

def classify_z(z):
    """Predics class of point based on its z-value (w^Tx). 
    """
    return 1 if z >= 0 else 0

def cart2pol(x, y):
    """Converts Cartesian coordinates to polar
    NOTE: NOT MY CODE
    http://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def c2p_a(A):
    """Custom coordinate transformation for matrix
    """
    new = []
    for ai in A:
        p, r = cart2pol(ai[1]-0.5, ai[2]-0.5)
        new.append([1, r, p])
    return np.array(new)

def d_line(x, w):
    """Decision line function
    """
    return -(w[0]+w[1]*x)/w[2]

def split(X, y):
    """Splits a data set based on classification
    """
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

def plot1D(inS0, outS0, inS1, outS1, inP, outP):
    """Plotting function for Task Task 2.1.3
    """
    fig, ax = plt.subplots()
    ax.scatter(inS0, outS0, marker='x', label='Training Data')
    ax.scatter(inS1, outS1, marker='x', label='Test Data')
    ax.plot(inP, outP, c='r', label='Hypothesis')
    ax.legend(loc='upper center', shadow=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("1d-linreg.png")
    plt.show()

def plot_error(errs, terrs):
    """Plotting function for cross-entropy error
    """
    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(errs))], errs, label='Training Data Error')
    ax.plot([i for i in range(len(terrs))], terrs, label='Test Data Error')
    ax.legend(shadow=True)
    plt.savefig("gd_error.png")
    plt.show()

def plot_class(X, y, Xtest, ytest, w, polar, fname):
    """Plotting function for classification tasks
    """
    x0, y0, x1, y1 = split(X, y)
    x0t, y0t, x1t, y1t = split(Xtest, ytest)
    d_line_x = [-3.4, 3.4]
    d_line_y = [d_line(x, w) for x in d_line_x]
    fig, ax = plt.subplots()
    if polar:
        ax.set_ylim([1.1, 0])
        ax.set_xlim([-3.5,3.5])
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\rho$')
    else:
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([-0.1,1.1])
        plt.xlabel('x')
        plt.ylabel('y')
    ax.scatter(x1, y1, marker='o', c='g', label='Class 1 Train')
    ax.scatter(x0, y0, marker='o', c='r', label='Class 0 Train')
    ax.scatter(x1t, y1t, marker='o', c='#005000', label='Class 1 Test')
    ax.scatter(x0t, y0t, marker='o', c='#500000', label='Class 0 Test')
    ax.plot(d_line_x, d_line_y, 'b--')
    ax.legend(shadow=True)
    plt.savefig(fname)
    plt.show()


def regression2D():
    """Solves Task 2.1.2
    Two dimentional linear regression
    """
    X, y = get_dataset(r2train)
    Xtest, ytest = get_dataset(r2test)
    w = ols(X, y)
    print('Resulting weights:')
    print(w)
    print('Error on training set: '+str(E_mse(X, y, w)))
    print('Error on test set: '+str(E_mse(Xtest, ytest, w)))

def regression1D():
    """Solves Task 2.1.3
    One dimentional linear regression
    """
    X, y = get_dataset(r1train)
    Xtest, ytest = get_dataset(r1test)
    w = ols(X, y)
    print('Error on training set: '+str(E_mse(X, y, w)))
    print('Error on test set: '+str(E_mse(Xtest, ytest, w)))
    inS = [x[1] for x in X]
    inP = np.array(sorted(inS))
    inS1 = [x[1] for x in Xtest]
    outP = [h(np.array([1, x]), w) for x in inP]
    plot1D(inS, y, inS1, ytest, inP, outP)

def classification1():
    """Solves Task 2.2.2
    Binary classification of linearly separable classes with error plots
    """
    X, y = get_dataset(c1train)
    Xtest, ytest = get_dataset(c1test)
    eta = 0.1
    n_rounds = 1000
    w = np.zeros(3)
    errs = []
    terrs = []
    for _ in range(n_rounds):
        w = logistic_gd(X, y, w, eta)
        errs.append(E_ce(X, y, w))
        terrs.append(E_ce(Xtest, ytest, w))
    plot_error(errs, terrs)
    plot_class(X, y, Xtest, ytest, w, False, 'class_1.png')

def classification2():
    """Task 2.2.3
    Binary classification of non-separable classes
    """
    X, y = get_dataset(c2train)
    Xtest, ytest = get_dataset(c2test)
    eta = 0.01
    n_rounds = 10000
    w = np.zeros(3)
    for _ in range(n_rounds):
        w = logistic_gd(X, y, w, eta)
    plot_class(X, y, Xtest, ytest, w, False, 'class_2.png')

def classification3():
    """Task 2.2.3
    Binary classification of non-separable classes,
    transformed with polar coordinates to be separable.
    """
    X, y = get_dataset(c2train)
    Xtest, ytest = get_dataset(c2test)
    X = c2p_a(X)
    Xtest = c2p_a(Xtest)
    eta = 0.1
    n_rounds = 1000
    w = np.zeros(3)
    for _ in range(n_rounds):
        w = logistic_gd(X, y, w, eta)
    plot_class(X, y, Xtest, ytest, w, True, 'class_2p.png')

def main():
    print('2D regression task (1)')
    print('1D regression task (2)')
    print('1st classification task (3)')
    print('2nd classification task (4)')
    print('2nd classification task with polar transformation (5)')
    selection = int(input('Select: '))
    if selection == 1:
        regression2D()
    elif selection == 2:
        regression1D()
    elif selection == 3:
        classification1()
    elif selection == 4:
        classification2()
    elif selection == 5:
        classification3()
    else:
        print('Not valid input')

if __name__ == '__main__':
    main()
