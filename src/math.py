# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:00:33 2019

@author: Mir

here we plot figures to show mathematical concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

# Height and width of a plot
plt.rcParams["figure.figsize"] = [3, 3]

def vector_plot(x, plot_name):
    
    """
    Plots a two-dimensional vector
    """
    
    fig = plt.figure()
    ax = plt.gca()
    
    origin = [0], [0]
    
    #ax.arrow(0, 0, x[0], x[1], head_width= 0.05, head_length=0.1)
    plt.quiver(*origin, x[0], x[1], angles='xy', scale_units='xy', scale=1,
               width=0.015)
    
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    # range of x and y axis
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    plt.xticks(np.arange(int(x_range[0]), int(x_range[1]) + 1, step=1))
    plt.yticks(np.arange(int(y_range[0]), int(y_range[1]) + 1, step=1))
    
    # axis major lines
    ax.xaxis.grid(which='major', linestyle='--')
    ax.yaxis.grid(which='major', linestyle='--')
    
    plt.draw()
    plt.show()
    
    fig.savefig(join('./figs/', plot_name + '.png'), format='png', dpi=500)

if __name__ == '__main__':
    
    x = np.array([2, 3])
    
    vector_plot(x, 'vector')