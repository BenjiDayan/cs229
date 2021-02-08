import numpy as np
import poisson.util as util
import os
import matplotlib.pyplot as plt
import time

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression()
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    y_test_pred = clf.predict(x_test)

    save_path_root = os.path.splitext(save_path)[0]
    # util.plot(x_test, y_test, clf.theta, save_path_root + '_fig.png')
    fig_name = save_path_root + '_fig.png'
    plt.figure()
    plt.scatter(y_test, y_test_pred, marker='x')
    plt.axline((0,0), slope=1)
    plt.xlabel('Observed counts')
    plt.ylabel('Predicted mean counts')
    plt.title('poisson glm fit predicted mean counts')
    plt.savefig(fig_name)

    np.savetxt(save_path, y_test_pred)
    np.savetxt(save_path_root + '_theta.txt', clf.theta)
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            print(f'step {i}:')
            prev_theta = self.theta.copy()
            self.step(x, y, self.step_size)
            theta_diff = (self.theta - prev_theta)
            size_theta_diff = theta_diff @ theta_diff
            print(f'size_theta_diff: {size_theta_diff}')
            if size_theta_diff < self.eps:
                break
        # *** END CODE HERE ***

    def step(self, x, y, lr):
        self.theta += lr * self.grad_log_likelihood(x, y, self.theta)
        print(f'new theta: {self.theta}')
        print(f'new log likelihood: {self.log_likelihood(x, y, self.theta)}')

    @staticmethod
    def log_likelihood(x, y, theta):
        """
        Args:
            x: (n x p array) matrix of n datapoints x_i
            y: (n, array) observed counts y_i
            theta: (p, array) assumed model of parameters eta_i = theta^T x_i

        Returns:
            scalar of log likelihood
        """
        eta = (x @ theta).reshape((-1,))  # (n,) array
        y_factorial = np.array([np.math.factorial(y_i) for y_i in y])
        l = -np.exp(eta).sum() + y.dot(eta) - np.log(y_factorial.astype(np.float)).sum()
        return l

    @staticmethod
    def grad_log_likelihood(x, y, theta):
        """
        Args:
            x: (n x p array) matrix of n datapoints x_i
            y: (n, array) observed counts y_i
            theta: (p, array) assumed model of parameters eta_i = theta^T x_i

        Returns:
            (p,) array, grad of log likelihood
        """
        eta = (x @ theta).reshape((-1,))  # (n,) array
        # (n,).dot(n x p) -> (p,)
        grad_l = -np.exp(eta).dot(x) + y.dot(x)

        return grad_l

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Having fit theta_MLE, have lambad_MLE = e^(theta^T x) = E(y|x)
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path=os.path.join('outputs', 'poisson_pred.txt'))
