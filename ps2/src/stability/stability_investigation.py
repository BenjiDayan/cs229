import os
import numpy as np

import util
from stability.stability_logreg import logistic_regression, calc_grad, theta_step
import matplotlib.pyplot as plt

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)

Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)

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

def scatter_dataset(X, Y, title, theta=None):
    os.makedirs('outputs', exist_ok=True)
    plt.figure()
    plt.scatter(X[Y == 1, 1], X[Y == 1, 2], c='r', label='Y==1')
    plt.scatter(X[Y == 0, 1], X[Y == 0, 2], c='g', label='Y==0')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)
    plt.legend()

    if not theta is None:
        slope = -1 / (theta[2] / theta[1])
        point = quick_solve(theta)
        plt.axline(point, slope=slope)
        theta_norm = 0.3 * theta[1:] / np.linalg.norm(theta[1:])
        plt.arrow(0.5, 0.5, theta_norm[0], theta_norm[1])

    plt.savefig(f'outputs/{title}.png')
    plt.show()


thetab = np.array([-107.04156569,  107.25200975,  107.08020705])
def h(theta, x):
    return 1/ (1 + np.exp(-x @ theta))

Yb_pred = (h(thetab, Xb) > 0.5).astype(np.float)

thetaa = np.zeros(3)
for i in range(30000):
    thetaa = theta_step(Xa, Ya, thetaa, i, 0.1)

scatter_dataset(Xa, Ya, 'Dataset_A_thetaa', thetaa)
scatter_dataset(Xb, Yb, 'Dataset_B_thetab', thetab)

diffb = Yb - h(thetab, Xb)
diffa = Ya - h(thetaa, Xa)
eg1 = diffb[np.abs(diffb) > 0.0001]
eg2 = diffa[np.abs(diffa) > 0.01]

# calc_grad(Xa, Ya, thetaa)
# Out[54]: array([-1.90078958e-14,  2.86289299e-14,  1.95224697e-14])
# calc_grad(Xb, Yb, thetab)
# Out[55]: array([-0.00537188,  0.00539149,  0.00537084])
# calc_grad(Xb, Yb, [-438.6524814 ,  441.25840523,  438.28730679])
# Out[64]: array([-0.0002178 ,  0.00021801,  0.00021764])

def get_grad_sizes(X, Y, theta, n, learning_rate):
    grad_sizes = []
    for i in range(n):
        grad_sizes.append(np.linalg.norm(calc_grad(X, Y, theta)))
        theta = theta_step(Xa, Ya, theta, i, learning_rate)
    return grad_sizes

gradsa = get_grad_sizes(Xa, Ya, np.zeros(3), 10000, 0.1)
gradsb = get_grad_sizes(Xb, Yb, np.zeros(3), 10000, 0.1)

plt.figure()
plt.plot(np.log(gradsa), label='dataset a')
plt.plot(np.log(gradsb), label='dataset b')
plt.legend()
plt.title('log |grad| against step')
plt.savefig('outputs/log_grad_size.png')