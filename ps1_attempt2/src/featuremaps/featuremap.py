import featuremaps.util as util
import os
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Normal equation J(theta) = (1/2) (X theta - y)^T (X theta - y)
        # grad J(theta) = X^T (X theta - y) = 0 when theta = (X^T X)^{-1} X y

        # X^T X may not be invertible, so instead np.linalg.solve(a, b) gives exact soln of ax = b.
        # (does this exist?)
        theta_MLE = np.linalg.solve(X.T @ X, X.T @ y)
        self.theta = theta_MLE
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return np.array([[intercept] + [x**j for j in range(1, k+1)] for intercept, x in X])
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return np.array(
            [[intercept] +
             [x ** j for j in range(1, k + 1)] +
             [np.sin(x)]
             for intercept, x in X]
        )
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return X @ self.theta
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        clf = LinearModel()
        create_feature_map = clf.create_poly if not sine else clf.create_sin
        x_train_hat = create_feature_map(k, train_x)
        clf.fit(x_train_hat, train_y)
        plot_x_hat = create_feature_map(k, plot_x)
        plot_y = clf.predict(plot_x_hat)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.title(os.path.splitext(os.path.basename(filename))[0])
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    os.makedirs('outputs', exist_ok=True)

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LinearModel()
    x_train_3 = clf.create_poly(3, x_train)
    clf.fit(x_train_3, y_train)

    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    x_test_3 = clf.create_poly(3, x_test)
    y_test_pred = clf.predict(x_test_3)

    save_path_root = 'outputs/degree_3_poly_fitting'
    # util.plot(x_test, y_test, clf.theta, save_path_root + '_fig.png')
    fig_name = save_path_root + '_fig.png'
    plt.figure()
    plt.scatter(x_test[:, 1], y_test, c='r', marker='x', label='y_test true func values')
    # plt.scatter(x_test[:, 1], y_test_pred, c='g', marker='o', label='predicted func values')
    x_range = np.linspace(x_test[:, 1].min(), x_test[:, 1].max(), num=100)
    x_range_3 = clf.create_poly(3, np.stack([np.ones(len(x_range)), x_range]).T)
    plt.plot(x_range, clf.predict(x_range_3), label='predicted func values')
    plt.xlabel('x')
    plt.ylabel('f(x) true and predicted')
    plt.title('degree 3 poly fitting')
    plt.legend()
    plt.savefig(fig_name)

    run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='outputs/multi_degree_plot.png')

    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename='outputs/multi_degree_with_sin_plot.png')

    run_exp(small_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='outputs/multi_degree_plot_small_dataset.png')
    run_exp(small_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename='outputs/multi_degree_plot_with_sin_small_dataset.png')

    # # Plot decision boundary on top of validation set set
    # x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    # y_test_pred = clf.predict(x_test)
    #
    # #    plot_decision_boundary(x_test, y_test, clf.theta, save_path)
    # util.plot(x_test, y_test, clf.theta, os.path.splitext(save_path)[0] + '_fig.png')
    # # Use np.savetxt to save predictions on eval set to save_path
    # np.savetxt(save_path, y_test_pred)
    # np.savetxt(save_path + '_theta', clf.theta)

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
