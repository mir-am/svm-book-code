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
    X_c2 = X[y == 0]
    
    # plot training samples of both classes
    plt.scatter(X_c1[:, 0], X_c1[:, 1], marker='^', color='blue', s=(50,))
    plt.scatter(X_c2[:, 0], X_c2[:, 1], marker='o', color='red',  s=(50,))
    
    # range of x and y axis
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    
    plt.ylabel(r'$x_{2}$')
    plt.xlabel(r'$x_{1}$')
    plt.xticks(np.arange(0, 1.1, step=1))
    plt.yticks(np.arange(0, 1.1, step=1))
    
    plt.tight_layout() # To fix axis labels not shown completely in the fig
    plt.show()
    
    print("Wanna save?... yes -> 1")
        
    if input() == '1':
    
        fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)



if __name__ == '__main__':
    
    #X, y = OR_dataset()
    X, y = XOR_dataset()
    plot_dataset(X, y, 'XOR-problem')
