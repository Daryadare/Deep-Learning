import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    tp, tn, fp, fn = 0, 0, 0, 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    for i in range(prediction.shape[0]):
        if prediction[i] and ground_truth[i]:
            tp += 1
        elif prediction[i] and not ground_truth[i]:
            fp += 1
        elif not prediction[i] and ground_truth[i]:
            fn += 1
        elif not prediction[i] and not ground_truth[i]:
            tn += 1

    accuracy = (tp + tn)/(tp + tn + fn + fp)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2 * precision * recall/(precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return np.sum(prediction == ground_truth)/len(prediction)
