import sys
import os
print(sys.path)
import numpy as np
import posonly.util as util

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, t_train)
    
    x_test, t_test = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    t_test_pred = clf.predict(x_test)
    util.plot(x_test, t_test, clf.theta, os.path.splitext(output_path_true)[0] + '_fig.png')
    np.savetxt(output_path_true, t_test_pred)
    np.savetxt(output_path_true + '_theta', clf.theta)
    
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_test, t_test = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    t_test_pred = clf.predict(x_test)
    util.plot(x_test, t_test, clf.theta, os.path.splitext(output_path_naive)[0] + '_fig.png')
    np.savetxt(output_path_naive, t_test_pred)
    np.savetxt(output_path_naive + '_theta', clf.theta)


    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted

    # We use alpha = E[ h(x) | y = 1] for perfect predictor h(x) = p(y=1 | x) and assumed p(t=1 | x) in {0,1} 0
    # uncertainty. For h we have a good estimate from logistic regression/ p(t=1 | x) = p(y=1 | x)/alpha

    # empirical estimate, using sample average h(x_i) for x_i: y_i = 1
    alpha_estimate = (1 / y_train.sum()) * clf.predict(x_train[y_train == 1]).sum()

    # H(x) := h(x)/alpha is our prediction for p(t=1 | x). Then H(x) = 1/2 <=> h(x) = alpha/2 = 1/(1 + e^(-theta^T x))
    # i.e. 2/alpha - 1 = e^(-theta^T x) so theta^T x = log(alpha/(2-alpha)
    # Then H(x) = 1/2 <=> phi^T x = 0 where phi_0 = theta_0 - log(alpha/(2-alpha)
    t_theta = clf.theta + np.array([-np.log(alpha_estimate / (2 - alpha_estimate)), 0., 0.])

    util.plot(x_test, t_test, t_theta, os.path.splitext(output_path_adjusted)[0] + '_fig.png')
    np.savetxt(output_path_adjusted, t_test_pred)
    np.savetxt(output_path_adjusted + '_theta', clf.theta)
    # *** END CODER HERE

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path=os.path.join('outputs', 'posonly_X_pred.txt'))
