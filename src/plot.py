# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:58:39 2019

@author: Mir, A.


This module contains code for plotting datasets, hyperplanes, and decision boundary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from os.path import join
from scipy.interpolate import make_interp_spline, BSpline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC


# Height and width of a plot
plt.rcParams["figure.figsize"] = [3, 3]


def OR_dataset():
    
    """
    Returns training samples and corresponding lables for the OR dataset
    """
    
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 1, 1, 0])

    return X, y    


def XOR_dataset():
    
    """
    Returns training samples and corresponding lables for the XOR dataset
    """

    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])

    return X, y 


def norm_dataset(cov_mat_c1, mean_vec_c1, cov_mat_c2, mean_vec_c2,
                 n_samples_c1, n_samples_c2):
    
    """
    It generates a random normal dataset with pre-defined covariace and mean
    """
    
    # Class +1
    X_1 = np.random.multivariate_normal(mean_vec_c1, cov_mat_c1, n_samples_c1)
    y_1 = np.ones(n_samples_c1, dtype=np.int)
    
    # Class -1
    X_2 = np.random.multivariate_normal(mean_vec_c2, cov_mat_c2, n_samples_c2)
    y_2 = np.ones(n_samples_c2, dtype=np.int) * -1
    
    return X_1, y_1, X_2, y_2


def make_dataset(X_1, y_1, X_2, y_2):
    
    """
    It merges data from two classes into one dataset
    """
    
    X = np.vstack((X_1, X_2))
    y = np.hstack((y_1, y_2))
    
    return X, y


def save_dataset(X, y, path):
    """
    It saves dataset into a CSV file.
    """
    
    df = pd.DataFrame(np.hstack((y.reshape(X.shape[0], 1), X)))
    df.to_csv(path, index=False, header=False)
      

def make_data_VC(plot_name):
    
    """
    It makes a 2-d dataset with 3 samples for showing VC dimension of the
    Perceptron.
    """
    
    #X = np.array([[2, 3], [1, 1], [3, 1]])
    X = np.array([[2, 1], [1, 3], [3, 3]])
    
    # Generate all combinations of labels for the data set which is 2^3
    comb_lables = np.array([list(i) for i in itertools.product([-1, 1],
                            repeat=3)])
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.07, wspace=0.1)
    
    for i in range(1, comb_lables.shape[0] + 1):
        
        ax = fig.add_subplot(2, 4, i)
        
        
        X_c1 = X[comb_lables[i - 1] == 1]
        X_c2 = X[comb_lables[i - 1] == -1]
        
        plt.scatter(X_c1[:, 0], X_c1[:, 1], marker='^', color='blue', s=(50,))
        plt.scatter(X_c2[:, 0], X_c2[:, 1], marker='o', color='red',  s=(50,))
        
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
    
        plt.xticks(np.arange(int(x_range[0]), int(x_range[1]) + 2, step=1))
        plt.yticks(np.arange(int(y_range[0]), int(y_range[1]) + 2, step=1))
        
        # No numbers on either axes
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        
    
    fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)
    
def plot_dataset(X, y, plot_name):
    
    """
    This plot data points in the feature space
    
    input:
        X: Training samples (NumPy array)
        y: class labels (NumPy array)
        plot_name: Name of the plot (str)
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Split training points into separate classes
    X_c1 = X[y == 1]
    X_c2 = X[y == -1]
    
    # plot training samples of both classes
    plt.scatter(X_c1[:, 0], X_c1[:, 1], marker='^', color='blue', s=(50,))
    plt.scatter(X_c2[:, 0], X_c2[:, 1], marker='o', color='red',  s=(50,))
    
    # range of x and y axis
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    
    plt.ylabel(r'$x_{2}$')
    plt.xlabel(r'$x_{1}$')
    plt.xticks(np.arange(int(x_range[0]) + 1, int(x_range[1]) + 1, step=1))
    plt.yticks(np.arange(int(y_range[0]) + 1, int(y_range[1]) + 1, step=1))
    
    plt.tight_layout() # To fix axis labels not shown completely in the fig
    plt.show()
    
    print("Wanna save?... yes -> 1")
    
    choice = int(input())
        
    if choice:
    
        fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)


def plot_3d_data(X, y, fig_name):
    """
    It plots a 3d-dimensional dataset.
    """
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # Split training points into separate classes
    X_c1 = X[y == 1]
    X_c2 = X[y == -1]
    
    ax.scatter(X_c1[:, 0], X_c1[:, 1], X_c1[:, 2], marker='^', color='blue',
               s=(50,))
    ax.scatter(X_c2[:, 0], X_c2[:, 1], X_c2[:, 2], marker='o', color='red',
               s=(50,))
    
    ax.set_xlabel(r'$x_{1}$', rotation=0)
    ax.set_ylabel(r'$x_{2}$', rotation=0)
    ax.set_zlabel(r'$x_{3}$', rotation=0)
    
    ax.set_yticks(np.arange(int(ax.get_ylim()[0]) , int(ax.get_ylim()[1] + 1),
                            step=44))
    
    plt.show()
    
    fig.savefig(join('./figs/', fig_name + '.png'), format='png', dpi=500)


def plot_quadratic(plot_name):
    
    """
    It plots a quadratic function
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    #  move left y-axis to center
    ax.spines['left'].set_position('center')
    #ax.spines['bottom'].set_position('center')
    
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    x = [-3, -2, -1, 0, 1, 2, 3]
    y = [i**2 for i in x]
    
    # Creating smooth curve
    xnew = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y, k=3)
    smooth_y = spl(xnew)
    
    plt.plot(xnew, smooth_y, color='black')
    
    # Plot constraint
    #plt.axvline(2, color='k', linestyle='dashed')
    
    # plot min point
    #plt.plot([0], [0], marker='o', color='k')
    
    # range of x and y axis
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    plt.xticks(np.arange(int(x_range[0]), int(x_range[1]) + 1, step=1))
    plt.yticks(np.arange(int(y_range[0]) + 1, int(y_range[1]) + 1, step=1))
    
    # axis major lines
    ax.xaxis.grid(which='major', linestyle='--')
    ax.yaxis.grid(which='major', linestyle='--')
    
    # Set margin to zero to make y-axis connected to x-axis
    plt.margins(0)
    plt.show()

    fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)
    
def read_data(filename):
    """
    It converts a CSV dataset to NumPy arrays.
    """
    
    df = pd.read_csv(filename)
    y = df.iloc[:, 0].values
    df.drop(df.columns[0], axis=1, inplace=True)
    X = df.values
    
    return X, y
    
    
def gen_data(n_samples):
    """
    Generating artificial dataset manually.
    """
    
    # Empty plot for manually generate data
    plt.plot()
    
    # Set Axis limit
    plt.ylim([-8, 8])
    plt.xlim([-8, 8])
    
    # Ginput for generating class 1 data
    c1_in = plt.ginput(n=n_samples, timeout=0, mouse_stop=2)
    
    # Convert to numpy array
    c1_data = np.asarray(c1_in)
    
    plt.scatter(c1_data[:, 0], c1_data[:, 1])
    plt.show()
    
    c2_in = plt.ginput(n=n_samples, timeout=0, mouse_stop=2)
    
    c2_data = np.asarray(c2_in)
    
    # Plot class 1  and -1 data points
    
    plt.scatter(c2_data[:, 0], c2_data[:, 1])
    plt.show()
    
    # Merge two classes
    X = np.row_stack((c1_data, c2_data))
    
    # Label of classes
    c1_label = np.ones((c1_data.shape[0], 1), dtype=np.int) # class 1 label
    
    c2_label = np.zeros((c2_data.shape[0], 1), dtype=np.int) # Class 2 label
    c2_label.fill(-1)
    
    # Merge lables
    y = np.row_stack((c1_label, c2_label)) 
    
    return X, y


def gen_mc_data(n_samples, n_class):
    """
    Generates multi-class classification data.
    """
    
    # Empty plot for manually generate data
    plt.plot()
    
    # Set Axis limit
    plt.ylim([-8, 8])
    plt.xlim([-8, 8])
    
    X_train = []
    y_train = []
    
    for c in range(n_class):
        
        c_i = np.array(plt.ginput(n=n_samples, timeout=0, mouse_stop=2))
        
        plt.scatter(c_i[:, 0], c_i[:, 1])
        plt.show()
        
        X_train.append(c_i)
        y_train = y_train + ([c] * n_samples)
        
    # Merege class sampes
    X = X_train[0]
    
    for i in range(len(X_train) - 1):
        
        X = np.row_stack((X, X_train[i+1]))
        
    return X, np.asarray(y_train)
    

def transform_2d_to_3d(X):
    """
    It transform a 2-d data to 3-d dimensional space.
    """
    
    X_3d = np.zeros((X.shape[0], 3), dtype=np.float)
    
    for i in range(X.shape[0]):
        
        X_3d[i, 0] = X[i, 0] ** 2
        X_3d[i, 1] = np.sqrt(2) * X[i, 0] * X[i, 1]
        X_3d[i, 2] = X[i, 1] ** 2
        
    return X_3d

def plot_func(plot_name, func):
    """
    It plots a single variable function.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    x = [-2, -1.5, -1, 0, 1, 1.5, 2]
    y = [func(i) for i in x]
    
    # Creating smooth curve
    xnew = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y, k=3)
    smooth_y = spl(xnew)
    
    plt.plot(xnew, smooth_y, color='black')
    
    # Plot constraint
    #plt.axvline(2, color='k', linestyle='dashed')
    
    # plot min point
    #plt.plot([0], [0], marker='o', color='k')
    
    # range of x and y axis
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    plt.xticks(np.arange(int(x_range[0]), int(x_range[1]) + 1, step=1))
    plt.yticks(np.arange(0, int(y_range[1]) + 1, step=1))
    
    # axis major lines
    ax.xaxis.grid(which='major', linestyle='--')
    ax.yaxis.grid(which='major', linestyle='--')
    
    plt.ylabel(r'$f(x)$')
    plt.xlabel(r'$x$')
    
    # Set margin to zero to make y-axis connected to x-axis
    #plt.margins(0)
    plt.tight_layout() # To fix axis labels not shown completely in the fig
    plt.show()

    fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)
    
    

def make_mesh(x, y, h=0.002):
    """
    Creates a mesh grid of points
    """
    
    step = 0.5
    
    x_min, x_max = x.min() - step, x.max() + step
    y_min, y_max = y.min() - step, y.max() + step
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    return xx, yy
    
    
def SVM_kernel_plot(X, y, fig_name):
    
    # mesh grid
    xx, yy = make_mesh(X[:, 0], X[:, 1])
    
    # Datapoints in inputspace
    data_points = np.c_[xx.ravel(), yy.ravel()]
    
    svm_model = SVC(kernel='poly', degree=1)
    svm_model.fit(X, y)
    
    # Predict class of data points
    z = svm_model.predict(data_points)
    print(z.shape)
    
    z = z.reshape(xx.shape)
    
    fig = plt.figure(1)
    
    axes = plt.gca()
        
    # plot decision boundary
    plt.contourf(xx, yy, z, levels=[-1, 0], colors='dimgray', alpha=0.8)
    
    # Split training points into separate classes
    X_c1 = X[y == 1]
    X_c2 = X[y == -1]
    
    # plot training samples of both classes
    plt.scatter(X_c1[:, 0], X_c1[:, 1], marker='^', s=(50,), c='b', cmap=plt.cm.coolwarm)
    plt.scatter(X_c2[:, 0], X_c2[:, 1], marker='o', s=(50,), c='r', cmap=plt.cm.coolwarm)
    #plt.scatter(X[:, 0], X[:, 1], marker='o', s=(50,), c=y, cmap=plt.cm.coolwarm)
    
    # Limit axis values
    axes.set_xlim(xx.min(), xx.max())
    axes.set_ylim(yy.min(), yy.max())
    
    # Show!
    plt.show()
    
    # Figure saved
    fig.savefig(fig_name, format='png', dpi=500)
    
    # Clear plot
    plt.clf()

    
    

if __name__ == '__main__':
    
     
    
#    # Cov matrices
#    c1_cov = np.array([[0, 0.2], [0.2, 0]])
#    c2_cov = np.array([[0, 0.2], [0.2, 0]])
#    
#    # Mean vectors
#    c1_mean = np.array([2, -1])
#    c2_mean = np.array([-0.5, 2])
#    
#    X_1, y_1, X_2, y_2 = norm_dataset(c1_cov, c1_mean, c2_cov, c2_mean, 8, 8)
#    X, y = make_dataset(X_1, y_1, X_2, y_2)
    
    #save = plot_dataset(X, y, 'non-linear-data')
        
#    if save:
#        
#        save_dataset(X, y, './2d-linear-data.csv')
    
#    X, y = gen_data(15)
#    save_dataset(X, y, './dataset/non-linear-data1.csv')

    #X, y = read_data('./dataset/2d-linear-data.csv') 
#    
#    X_3d = transform_2d_to_3d(X)
#    plot_3d_data(X_3d, y.astype('int'), '2d-3d-tansform')

    
    #plot_quadratic('quadratic1')
    # lambda i :  i**6 + i**4 + i**2 + 2
    # lambda i :  i**2
    
    #plot_func('weak-convex-func', lambda i : i**2 + 2)
    
    #SVM_kernel_plot(X, y, 'poly-SVM-poly-d1.png')
    
    X, y = gen_mc_data(7, 3)
    save_dataset(X, y, './dataset/mc-data.csv')
    