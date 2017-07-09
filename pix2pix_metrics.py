import numpy as np

def error(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return np.abs(label - (pred > 0.5)).sum() / label.shape[0]

def accuracy(label, pred):
    # Accuracy is also used as a statistical measure of how well a
    # binary classification test correctly identifies or excludes a condition.
    # That is, the accuracy is the proportion of true results
    # (both true positives and true negatives) among the total number of cases examined[7]
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


def entropy(label, pred):
    # Entropy on the other hand is a measure of impurity (the opposite).
    # It is defined for a binary class with values a/b as:
    # Entropy = - p(a)*log(p(a)) - p(b)*log(p(b))
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    pred = pred.ravel()
    label = label.ravel()
    return -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12)).mean()