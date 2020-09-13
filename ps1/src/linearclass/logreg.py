import numpy as np
from linearclass import util
import time


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
    LogRegger = LogisticRegression()
    LogRegger.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***

def J(theta, x, y):
    """avg empirical loss over x and y - i.e -1/n sum log likelihood

    Args:
        x: Training example inputs. Shape (n_examples, dim).
        y: Training example labels. Shape (n_examples,).
    """

    return (-1/y.shape[0]) * (
        y @ np.log(g(x @ theta)) +
        (1-y) @ np.log(1 - g(x @ theta))
    )

def grad_J(theta, x, y):
    """gradient of avg empirical loss over x and y - i.e -1/n sum log likelihood

    Args:
        x: Training example inputs. Shape (n_examples, dim).
        y: Training example labels. Shape (n_examples,).
    """
    return (-1/y.shape[0]) * (
        np.multiply(y , 1 - g(x @ theta)) @  x +
        np.multiply(1 - y ,g(x @ theta)) @ x
    )

def g(z):
    eps = 1e-7
    return np.minimum(np.maximum(1 / (1 + np.exp(-z)), eps), 1-eps)
    #return 1 / (1 + np.exp(-z))

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

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[-1])
        for step_i in range(self.max_iter):
            grad = grad_J(self.theta, x, y)
            loss = J(self.theta, x, y)
            print(f'step: {step_i}, loss: {loss}, theta: {self.theta}')
            # print(f'delta: {self.step_size * grad}')
            old_theta = self.theta.copy()
            self.theta -= self.step_size * grad
            # print(old_theta)
            # print(self.theta)
            # print(np.sum(np.abs(self.theta - old_theta)))
            # print(np.sum(np.abs(self.theta - old_theta)) < self.eps)
            if np.sum(np.abs(self.theta - old_theta)) < self.eps:
                break

            time.sleep(0.)
        # *** END CODE HERE ***


    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

if __name__ == '__main__':
    print('hi!')
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')
#
#     main(train_path='ds2_train.csv',
#          valid_path='ds2_valid.csv',
#          save_path='logreg_pred_2.txt')
