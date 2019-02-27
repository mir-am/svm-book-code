# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:58:39 2019

@author: Mir, A.


This module contains code for plotting datasets, hyperplanes, and decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

# Height and width of a plot
#plt.rcParams["figure.figsize"] = [3, 3]


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
        
    if input() == '1':
    
        fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)



if __name__ == '__main__':
    
    # Cov matrices
    c1_cov = np.array([[1, 0], [0, 1]])
    c2_cov = np.array([[0.5, 0], [0, 0.5]])
    
    # Mean vectors
    c1_mean = np.array([3, 3])
    c2_mean = np.array([0, 0])
    
    X_1, y_1, X_2, y_2 = norm_dataset(c1_cov, c1_mean, c2_cov, c2_mean, 20, 20)
    X, y = make_dataset(X_1, y_1, X_2, y_2)
    plot_dataset(X, y, 'LinearDataset')
