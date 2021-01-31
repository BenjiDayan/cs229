import numpy as np
import linearclass.util as util
import time
import os
from general_utils import sigmoid, quick_solve, plot_decision_boundary

#  x_train, y_train = util.load_dataset('ds1_train.csv', add_intercept=True)
# (800, 3), (800,)

eps = 1e-6

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    # Plot decision boundary on top of validation set set
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    y_test_pred = clf.predict(x_test)
    
#    plot_decision_boundary(x_test, y_test, clf.theta, save_path)
    util.plot(x_test, y_test, clf.theta, os.path.splitext(save_path)[0] + '_fig.png')
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_test_pred)
    np.savetxt(save_path + '_theta', clf.theta)
    # *** END CODE HERE ***

def g(z):
    pass

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y, method='newton'):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
            method: (str) 'newton' or 'euler'
        """
        # *** START CODE HERE ***
        if method == 'newton':
            step_func = self.step_newton
        elif method == 'euler':
            step_func = self.step_euler
        else:
            raise Exception('method should be "newton" or "euler"')
            
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        for i in range(self.max_iter):
            print(f'Step {i}:')
            prev_theta = self.theta.copy()
            step_func(x, y)
            theta_diff = self.theta - prev_theta
            theta_diff_size = np.abs(theta_diff).sum()
            print(f'theta_diff: {theta_diff} has ||theta_diff||1 = {theta_diff_size}')
            if theta_diff_size < self.eps:
                print(f'theta_diff: {theta_diff} has ||theta_diff||1 < eps={self.eps}')
                break
            
#            time.sleep(1)
        # *** END CODE HERE ***
        
    def step_euler(self, x, y):
        """
        Euler step theta -> theta + alpha grad l(theta) where alpha > 0,
        l is log likelihood. I.e. seeking to maximise likelihood
        """
        grad_theta = self.grad_log_likelihood(x, y)
        log_likelihood = self.log_likelihood(x, y)
        print(f'grad_theta: {grad_theta}')
        print(f'log_likelihood before step: {log_likelihood}')
        self.theta += grad_theta * self.step_size
        print(f'theta: {self.theta}')
    
    
    def step_newton(self, x, y):
        """To solve f(theta)=0, step theta -> theta - f/f'
        So to maximise l(theta), solve for l'(theta)=0, i.e. 
        theta -> theta - l'/l''.
        For vectors, theta -> theta - (H^-1) grad_theta l(theta)
        """
        grad_theta = self.grad_log_likelihood(x, y)
        hessian = self.hessian_log_likelihood(x, y)
        log_likelihood = self.log_likelihood(x, y)
        print(f'grad_theta: {grad_theta}')
        print(f'log_likelihood before step: {log_likelihood}')
        self.theta -= np.linalg.inv(hessian) @ grad_theta
        print(f'theta: {self.theta}')
        
        
    def log_likelihood(self, x, y, theta=None):
        """This is the loss: minus average logistic likelihood = l(theta)"""
        theta = theta if not theta is None else self.theta
        n = y.shape[0]
        loss_vec = (-1/n) * (
            y * np.log(sigmoid(x, theta) + eps) + 
            (1-y) * np.log(1 - sigmoid(x, theta) + eps)
        )
        return -loss_vec.sum()
    
    def grad_log_likelihood(self, x, y, theta=None):
        theta = theta if not theta is None else self.theta
        n = y.shape[0]
        stacked = np.stack([y - sigmoid(theta, x)]*x.shape[-1])
        stacked = stacked.transpose(list(range(1, len(stacked.shape))) + [0])
        return ((1/n)* (x) * stacked).sum(axis=0)
    
    def hessian_log_likelihood(self, x, y, theta=None):
        theta = theta if not theta is None else self.theta
        n = y.shape[0]
        h = sigmoid(x, theta)
        scalars = (-1/n) * h * (1-h)
        out = np.zeros((x.shape[1], x.shape[1]))
        for x_row, scalar in zip(x, scalars):
            x_row_2d = x_row.reshape((-1, 1))
            out += scalar * x_row_2d @ x_row_2d.T
        return out
    
        
    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return sigmoid(x, self.theta)
        # *** END CODE HERE ***
        
    
    

#clf = LogisticRegression(theta_0 = np.array([1.,2.,3.]))
# actually this theta_0 is terrible for newton method ;( don't know why

#clf = LogisticRegression()
#
#train_path='ds1_train.csv'
#valid_path='ds1_valid.csv'
#x_train, y_train = util.load_dataset(train_path, add_intercept=True)
#
## *** START CODE HERE ***
## Train a logistic regression classifier
#clf = LogisticRegression()
#clf.fit(x_train, y_train)
#
## Plot decision boundary on top of validation set set
#x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
#y_test_pred = clf.predict(x_test)
#
#plot_decision_boundary(x_test, y_test, clf.theta)




if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1_v2.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2_v2.txt')
