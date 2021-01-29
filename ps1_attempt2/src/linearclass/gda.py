import numpy as np
import os
import util
from util import sigmoid, quick_solve, plot_decision_boundary
import matplotlib.pyplot as plt


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    y_test_pred = clf.predict(x_test[:, 1:])
    
#    plot_decision_boundary(x_test, y_test, clf.theta, save_path)
    plot(x_test, y_test, clf.theta, os.path.splitext(save_path)[0] + '_fig.png')
    
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_test_pred)
    base, ext = os.path.splitext(save_path)
    theta_save_path = base + '_theta' + ext
    np.savetxt(theta_save_path, clf.theta)
    
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # maximum likelihood estimates:
        phi = np.sum(y)/y.shape[0]
        x0, x1 = x[np.argwhere(y==0.).squeeze()], x[np.argwhere(y==1.).squeeze()]
        mu0, mu1 = x0.mean(axis=0), x1.mean(axis=0)
        sigma = (
            (x0 - mu0).T @ (x0 - mu0) +
            (x1 - mu1).T @ (x1 - mu1)
        ) / y.shape[0]
        sigma_inv = np.linalg.inv(sigma)
        # Write theta in terms of the parameters
        # PS1 has p(y=1| x; phi; mu0; mu1; sigma) = 1/(1 + exp(-theta^T x -theta_0))
        theta = (mu0 - mu1).T @ sigma_inv
        theta0 = 0.5 * (mu1.T @ sigma_inv @ mu1 -
                        mu0.T @ sigma_inv @ mu0 +
                        np.log(phi/(1-phi)))
        self.theta = np.array([theta0] + list(theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x_with_intercept = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        return sigmoid(x_with_intercept, self.theta)
        
        # *** END CODE HERE
        
def transforming_stuff(train_path, valid_path, save_path):
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_train2 = np.stack([x_train[:, 0], np.log(x_train[:, 1])]).T
    
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train2, y_train)
    # Plot decision boundary on validation set
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    y_test_pred = clf.predict(x_test[:, 1:])
    

    x, y = x_test, y_test
    plt.figure()
    plt.plot(x[y==0, -2], x[y==0, -1], 'bx', linewidth=2)
    plt.plot(x[y==1, -2], x[y==1, -1], 'go', linewidth=2)
    
    x1_min, x1_max = x[:, -2].min(), x[:, -2].max()
    x_pts = np.arange(x1_min, x1_max, (x1_max-x1_min)/100)
    theta0, theta1, theta2 = clf.theta
    y_pts = np.exp((-1/theta2) * theta1*x_pts + theta0)
    plt.plot(x_pts, y_pts)
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    if save_path:
        plt.savefig(os.path.splitext(save_path)[0] + '_fig.png')
    
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_test_pred)
    base, ext = os.path.splitext(save_path)
    theta_save_path = base + '_theta' + ext
    np.savetxt(theta_save_path, clf.theta)

# Use np.savetxt to save outputs from validation set to save_path
# *** END CODE HERE ***

if __name__ == '__main__':
    train_path = 'ds1_train.csv'
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
    
    transforming_stuff('ds1_train.csv', 'ds1_valid.csv', 'gda_pred_1_logtrans.txt')
    
