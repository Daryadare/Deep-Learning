import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    tp, tn, fp, fn = 0, 0, 0, 0

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

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return np.sum(prediction == ground_truth)/len(prediction)
