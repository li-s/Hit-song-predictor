import sklearn
import numpy as np


def get_confusion_matrix(predicted_labels, actual_labels):
    assert len(predicted_labels) == len(actual_labels)
    data_size = actual_labels.shape[0]
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for row in range(data_size):
        actual = int(actual_labels[row])
        predicted = int(predicted_labels[row])

        if actual == 0 and predicted == 0:
            # true negative
            true_neg += 1
        elif actual == 1 and predicted == 1:
            # true positive
            true_pos += 1
        elif actual == 0 and predicted == 1:
            # false positive
            false_pos += 1
        elif actual == 1 and predicted == 0:
            # false negative
            false_neg += 1

    return true_pos, true_neg, false_pos, false_neg


def evaluate_accuracy(true_pos, true_neg, false_pos, false_neg):
    mean_accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    return mean_accuracy * 100


def evaluate_f1(true_pos, true_neg, false_pos, false_neg):
    f1 = true_pos / (true_pos + 0.5 * (false_pos + false_neg))
    return f1 * 100


def evaluate_recall(true_pos, true_neg, false_pos, false_neg):
    recall = true_pos / (true_pos + false_neg)
    return recall * 100


def evaluate_precision(true_pos, true_neg, false_pos, false_neg):
    precision = true_pos / (true_pos + false_pos)
    return precision * 100


def get_overall_evaluation(predicted_labels, actual_labels, X_train, y_train, model=None):
    if not (model is None):
        scorings = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
        scores = sklearn.model_selection.cross_validate(model, X_train, y_train, scoring=scorings, cv=10)
        print('Validation scores using 10-fold cross validation on training data:')
        print('  Precision score: %.2f' % (np.mean(scores['test_precision']) * 100) + '%')
        print('  Recall score: %.2f' % (np.mean(scores['test_recall']) * 100) + '%')
        print('  F1 score: %.2f' % (np.mean(scores['test_f1']) * 100) + '%')
        print('  Mean accuracy score: %.2f' % (np.mean(scores['test_accuracy']) * 100) + '%')
        print('  ROC AUC: %.2f' % (np.mean(scores['test_roc_auc']) * 100) + '%')

    print()
    print('Test scores using test data:')
    true_pos, true_neg, false_pos, false_neg = get_confusion_matrix(predicted_labels, actual_labels)
    print('  Precision score: %.2f' % evaluate_precision(true_pos, true_neg, false_pos, false_neg) + '%')
    print('  Recall score: %.2f' % evaluate_recall(true_pos, true_neg, false_pos, false_neg) + '%')
    print('  F1 score: %.2f' % evaluate_f1(true_pos, true_neg, false_pos, false_neg) + '%')
    print('  Mean accuracy score: %.2f' % evaluate_accuracy(true_pos, true_neg, false_pos, false_neg) + '%')
