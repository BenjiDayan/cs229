import collections

import numpy as np
import re

import itertools
import spam.util as util
from spam.svm import *
from collections import Counter
from functools import reduce

from dataclasses import dataclass

@dataclass
class NaiveBayesMulinomialProbs:
    message_spam_prob: float
    word_spam_probs: np.ndarray

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return [x.lower() for x in re.findall(r"[\w']+", message)]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counter = Counter(reduce(lambda a, b: a +b, map(get_words, messages)))
    i = 0
    output = {}
    for k, v in counter.items():
        if v >= 5:
            output[k] = i
            i += 1
    return output
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    output = []
    print(output)
    for message in messages:
        words = Counter(get_words(message))
        output.append([0 if not word in words else words[word] for word in word_dictionary])
    print(len(output))
    foo = np.array(output)
    print(foo.shape)
    return foo
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model give a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    num_messages = matrix.shape[0]
    num_words = matrix.shape[1]
    word_probs = np.zeros((2, num_words))

    num_spam_messages = labels.sum()
    num_spam_words = matrix[labels==1].sum()
    num_non_spam_words = matrix[labels==0].sum()

    spam_prob = num_spam_messages / num_messages
    word_probs[0, :] = (1 + matrix[labels==0].sum(axis=0)) / (num_non_spam_words + num_words)
    word_probs[1, :] = (1 + matrix[labels == 1].sum(axis=0)) / (num_spam_words + num_words)

    return NaiveBayesMulinomialProbs(spam_prob, word_probs)
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model: NaiveBayesMulinomialProbs, matrix: np.ndarray):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix (nxm) : A numpy array containing word counts. n is number of messages, and m is size of dictionary

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    spam_prob, word_probs = model.message_spam_prob, model.word_spam_probs
    log_word_probs = np.log(word_probs)  # [ [log phi_0|y=0, log phi_1|y=0, log phi_2|y=0, ...], [log phi_0|y=1, ... ]]
    matrix_cond0_probs = spam_prob * np.exp(np.dot(matrix, log_word_probs[0, :]))
    matrix_cond1_probs = (1 - spam_prob) * np.exp(np.dot(matrix, log_word_probs[1, :]))

    message_spam_probs = matrix_cond1_probs / (matrix_cond0_probs + matrix_cond1_probs)
    return message_spam_probs > 0.5

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model: NaiveBayesMulinomialProbs, dictionary: dict):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    word_spam_probs = model.word_spam_probs
    word_spam_indication_score = np.log(word_spam_probs[1, :] / word_spam_probs[0, :])
    top_num = 5
    words = list(dictionary.keys())
    return [words[i] for i in itertools.islice(reversed(np.argsort(word_spam_indication_score)), top_num)]
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix (nxm): The word counts for the training data. n is num messages, m is size dictionary.
        train_labels (n): The spam or not spam labels for the training data. n is num messages
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***#
    def simulated_annealing(f, starting_val=1.0, n=10000, sigma=1.0, jump_prob=0.5, prob_shrink_rate=0.9):
        prev_out = f(starting_val)
        for _ in range(n):
            new_val = starting_val + np.random.normal(scale=sigma)
            new_out = f(new_val)
            if new_out > prev_out or np.random.rand() < jump_prob:
                starting_val = new_val
                prev_out = new_out

            jump_prob *= prob_shrink_rate

        return starting_val

    def get_accuracy(radius):
        state = svm_train(train_matrix, train_labels, radius)
        val_preds = svm_predict(state, val_matrix, radius)
        val_accuracy = np.mean(val_preds == val_labels)
        return val_accuracy

    return radius_to_consider[np.argmax(map(get_accuracy, radius_to_consider))]

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
    # can quick check predictions: [test_messages[i] for i, x in enumerate(naive_bayes_predictions > 0.5) if x]
    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    radius = 1.0
    state = svm_train(train_matrix, train_labels, radius)
    svm_preds = svm_predict(state, test_matrix, radius)
    svm_accuracy = np.mean(list(map(bool, svm_preds)) == test_labels)
    disagree = [i for i, (x, y) in enumerate(zip(svm_preds, naive_bayes_predictions)) if bool(x) != y]

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
