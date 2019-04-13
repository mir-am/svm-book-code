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
    plt.xticks(np.arange(int(x_range[0]), int(x_range[1]) + 1, step=1))
    plt.yticks(np.arange(int(y_range[0]), int(y_range[1]) + 1, step=1))
    
    plt.tight_layout() # To fix axis labels not shown completely in the fig
    plt.show()
    
    print("Wanna save?... yes -> 1")
    
    choice = int(input())
        
    if choice:
    
        fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)


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
    
def gen_data(n_samples):
    """
    Generating artificial dataset manually.
    """
    
    # Empty plot for manually generate data
    plt.plot()
    
    # Set Axis limit
    plt.ylim([0, 3])
    plt.xlim([0, 3])
    
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
#    save = plot_dataset(X, y, 'LinearDataset3')
#        
#    if save:
#        
#        save_dataset(X, y, './2d-linear-data.csv')
    
    X, y = gen_data(10)
    save_dataset(X, y, './soft-margin-data.csv')    
    
    #plot_quadratic('quadratic')