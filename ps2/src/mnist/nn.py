import pickle

import numpy as np
import matplotlib.pyplot as plt
import argparse

from typing import List, Tuple
from tqdm import tqdm

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # We want to avoid computing things like np.exp(10000) = inf and np.exp(-10000) = 0.0
    # We want e.g. e^10000 / (e^10000 + e^10010 + e^10) ~= 1/(1 + e^10 + 0.0) = 4.5398 10^{-5}
    # it's ok if e^10 / (e^10000 + e^10010 + e^10) ~= 0.0
    #
    # What's causing overflow is the annoyingly big x_max = max(x_i). So we divide top and bottom of fraction by this.
    # softmax(x)_i = (e^{x_i}) / (sum_j e^{x_j}) = (e^{x_i - x_max}) / (sum_j e^{x_j - x_max})

    x_max = x.max(axis=1).reshape(-1, 1)
    x2 = np.exp(x - x_max)
    return x2 / x2.sum(axis=1).reshape(-1, 1)

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    return 1 / (1 + np.exp(-x))


def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # z^[l] = W^[l] a^[l-1] + b^[l]; a^[l] = g_l (z^[l]). Probably g_l are all the same e.g. sigmoid except
    # last layer g_n = softmax.
    # Xavier/He initialization: w^[l] ~ N(0, sqrt(2 / n^[l] + n^[l-1])) where n^[l] is # of neurons in lth layer.
    (W_1, b_1), (W_2, b_2) = initialize_layers([input_size, num_hidden, num_output])
    # return {
    #     'W_1': W_1, 'b_1': b_1,
    #     'W_2': W_2, 'b_2': b_2
    # }
    return [[W_1, b_1], [W_2, b_2]]


def initialize_layers(n_layers: List[int]):
    assert len(n_layers) >= 1
    output_params = []
    n_layers = list(reversed(n_layers))
    n_in = n_layers.pop()
    while len(n_layers) > 0:
        n_out = n_layers.pop()
        output_params.append(initialize_layer(n_in, n_out))
        n_in = n_out
    return output_params

def initialize_layer(n_in, n_out):
    """where z^[l-1] is a n_in vec, z^[l] = W^[l] a^[l-1] + b^[l] is a n_out vec"""
    std = np.sqrt(2 / (n_in + n_out))
    # don't start b as np.random.normal(size=(n_out,), scale=std) for whatever reason
    W, b = np.random.normal(size=(n_out, n_in), scale=std), np.zeros((n_out,))
    return W, b

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """

    activations = []
    for i, (W, b) in enumerate(params):
        activations.append(data)
        data = data @ W.T + b  # (B x N_i) @ (N_i x N_{i+1})
        g = sigmoid if i < len(params)-1 else softmax
        data = g(data)

    output = data
    # cross entropy loss
    avg_loss = -(1/len(labels)) * (labels * np.log(output)).sum()
    return [activations, output, avg_loss]


def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propagation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """

    activations, output, avg_loss = forward_prop_func(data, labels, params)
    # grad_{z^n} CE_k (y, y_hat) = y^k_hat - y^k for kth label  vec y^k and prediction y^k_hat
    # This is a length c vector where c is the number of classes

    # Focus on a single label/prediction pair for now (drop k index)
    # Writing d^n = grad_{z^n} CE (y, y_hat), we get grad_{W^n_ij} CE = (d^n)_i a^n_j
    # and grad_{b^n_j} CE = (d^n)_j

    # So we'll return gradients = (1/B) [[sum_k grad_{W^1_ij} CE_k, sum_k grad_{b^1_ij} CE_k], [,]_2, ..., [,]_n]
    # Where B is the batch size, and sum_k is over each datapoint in the batch

    num_layers = len(params)
    # e.g. a0 (input layer) -> (w1) -> z1,a1 -> (w2) -> z2,a2 -> (w3) z3, o3 (o3 is final output, softmax)
    # num_layers is 3 here. Also 3 activations.
    assert len(activations) == num_layers
    batch_size, num_classes = labels.shape
    d = (output - labels)/batch_size # (batch_size x num_classes) matrix with kth row y^k_hat - y^k
    gradients = [[]]*num_layers
    # gradients[num_layers-1] = (output - labels)/batch_size
    for i in range(num_layers-1, -1, -1):
        # batch_size x N_i x 1 @ 1 x N_{i-1} = B x N_i x N_{i-1}
        activation = activations[i]

        d_new_shape = [batch_size, d.shape[1], 1]
        a_new_shape = [batch_size, 1, activation.shape[1]]
        W_grad = d.reshape(d_new_shape) @ activation.reshape(a_new_shape)
        W_grad = W_grad.sum(axis=0)  # sum over datapoints in batch
        b_grad = d.sum(axis=0)
        gradients[i] = [W_grad, b_grad]

        # d_{i-1} = (d_i)_k * (del z^i_k)/(del z^{i-1}_j) = (d_i)_k * W_kj g'(z_j) = (d_i)_k W_kj a_j (1 - a_j)
        W, _b = params[i]
        # (batch_size x N_i) x (N_i x N_{i-1}) then eltwise multiplication with (batch_size x N_{i-1}
        d = (d @ W) * activation * (1 - activation)

    return gradients


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propagation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    gradients = backward_prop(data, labels, params, forward_prop_func)
    # regularization term acts to decrease ||W^n||^2
    return [[W_grad + 2 * reg * W, b_grad] for (W_grad, b_grad), (W, _b) in zip(gradients, params)]

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """
    def update_params(
            params: List[Tuple[np.ndarray, np.ndarray]],
            neg_gradients: List[Tuple[np.ndarray, np.ndarray]],
            learning_rate: float
    ) -> None:
        """both params, gradients are a lists of [[W_1, b_1], [W_2, b_2], ... [W_n, b_n]]
        where W_i, b_i are np.ndarray. This func just upates the params in place
        """
        for i in range(len(params)):
            params[i][0] -= learning_rate * neg_gradients[i][0]
            params[i][1] -= learning_rate * neg_gradients[i][1]

    n = len(train_data)
    num_batches = np.math.ceil(n/batch_size)
    for batch_i in range(num_batches):
        print(f'batch {batch_i} of {num_batches}')
        i1, i2 = batch_size * batch_i, min(n, batch_size * (batch_i + 1))
        x, y = train_data[i1:i2], train_labels[i1:i2]
        neg_gradients = backward_prop_func(x, y, params, forward_prop_func)

        update_params(params, neg_gradients, learning_rate)

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in tqdm(range(num_epochs)):
        print(f'epoch {epoch} of {num_epochs}')
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )
    pickle_out_dict = {
        'params': params,
        'cost_train': cost_train,
        'cost_dev': cost_dev,
        'accuracy_train': accuracy_train,
        'accuracy_dev': accuracy_dev,
    }

    with open(f'{name}_pickle_out', 'wb') as file:
        pickle.dump(pickle_out_dict, file)

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    # train_data, train_labels = read_data('./minis/images_train.csv', './minis/labels_train.csv')
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    n_train = train_labels.shape[0]  # normally 60000
    dev_prop = 1/6
    n_dev = int(n_train * dev_prop)  # normally 10000
    p = np.random.permutation(n_train)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:n_dev,:]
    dev_labels = train_labels[0:n_dev,:]
    train_data = train_data[n_dev:,:]
    train_labels = train_labels[n_dev:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    # test_data, test_labels = read_data('./minis/images_test.csv', './minis/labels_test.csv')
    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    # TODO remove and the bit below
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)

    # return reg_acc
    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
