# Important note: you do not have to modify this file for your homework.

import numpy as np
np.random.seed(123)


def train_and_predict_svm(train_matrix, train_labels, test_matrix, radius):
    """Train an SVM model and predict the resulting labels on a test set.

    Args:
        train_matrix: A numpy array containing the word counts for the train set
        train_labels: A numpy array containing the spam or not spam labels for the train set
        test_matrix: A numpy array containing the word counts for the test set
        radius: The RBF kernel radius to use for the SVM

    Return:
        The predicted labels for each message
    """
    model = svm_train(train_matrix, train_labels, radius)
    return svm_predict(model, test_matrix, radius)


def svm_train(matrix, category, radius):
    """

    Args:
        matrix: A numpy array containing the word counts for the train set
        category: A numpy array containing the spam or not spam labels for the train set
        radius: The RBF kernel radius to use for the SVM

    Returns:

    """
    state = {}
    M, N = matrix.shape
    Y = 2 * category - 1

    # K(x,z) = exp(-||x-z||^2 / (2 sigma^2))
    matrix = 1. * (matrix > 0)  # [0, 2, 5, 0, 0, 1, ...] -> [0., 1., 1., 0., 0., 1., ...]
    squared = np.sum(matrix * matrix, axis=1)  # matrix_ij = (x_i)_j. squared_i = (x_i)^T (x_i)
    gram = matrix.dot(matrix.T)
    # ||x-z||^2 = ||x||^2 + ||z||^2 - 2 x^T z
    # so ||x_i - x_j||^2 = squared_i + squared_j - 2 * gram_ij

    # gram_ij = matrix_ik matrix_jk = (x_i)_k (x_j)_k = x_i^T x_j.
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (radius ** 2)))
    # # K_ij = exp(-||x_i-x_j||^2 / (2 sigma^2))

    # in notes we have e.g. w^T x = sum_i alpha_i y_i <x_i, x>
    # we'll define  a_i = alpha_i y_i. Then w^T x = sum_i a_i <x_i, x>
    # Then for fixed x_i have y_i w^T x_i = sum_j a_j <x_j, x_i> = sum_j a_j <x_i, x_j>
    # So define margin = y_i w^T x_i = Y[i] * np.dot(K[i, :], a)
    # We always want margin to be large and positive.
    # problem is phrased as min 1/2 ||w||^2 s.t. y_i (w^T x_i) >= 1 for i=1,...,M
    # Non separable case allowed through regularization. Now y_i (w^T x_i) >= 1 - z_i for i=1,...,M
    # and objective min 1/2 ||w||^2 + C sum z_i.
    # Hence if margin < 1 we incur a penalty z_i

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 10

    alpha_avg = 0
    ii = 0
    while ii < outer_loops * M:
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if margin < 1:
            grad -= Y[i] * K[:, i]
        alpha -= grad / np.sqrt(ii + 1)
        alpha_avg += alpha
        ii += 1

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    return state


def svm_predict(state, matrix, radius):
    M, N = matrix.shape

    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    # K_ij = exp(||x_i - x'_j||^2 / (2 sigma^2)) where x'_j is the jth training vec, and x_i is the ith test vec.
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (radius ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = (1 + np.sign(preds)) // 2

    # normally w^T x = sum over training j alpha_j y_j K_ij but here we have sum K_ij alpha_avg_j.
    # what is alpha_avg?

    # 7

    return output
