# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x, theta):
    """
    x (n x p matrix)
    theta (p vector)
    """
    return 1/ (1 + np.exp(-x @ theta.T))

def quick_solve(theta):
    w1, w2 = theta[1:]
    b = theta[0]
    if w1 != 0:
        return np.array([-b/w1, 0])
    elif w2 != 0:
        return np.array([0, -b/w2])
    elif b == 0:
        return np.array([0, 0])
    else:
        raise Exception('w1, w2 both zero yet b is non zero')


def plot_decision_boundary(x, y, theta=None, save_path='', intercept=True):
    plt.figure()
    start_i = 1 if intercept else 0
    plt.scatter(x[y==0, start_i], x[y==0, start_i + 1], color='r')
    plt.scatter(x[y==1, start_i], x[y==1, start_i + 1], color='b')
    
    if not theta is None:
        slope = -1/(theta[2]/theta[1])
        point = quick_solve(theta)
        plt.axline(point, slope=slope)
    
    if save_path:
        plt.savefig(os.path.splitext(save_path)[0] + '_fig.png')
    
    plt.show()