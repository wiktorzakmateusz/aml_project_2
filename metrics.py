import numpy as np
from sklearn.metrics import precision_score


# for a fixed number of features, this is the measure we want to maximize
# so we want to adapt models, parameters and architectures to obtain as large score as possible

# but it is not the ultimate function to optimize
# the final function takes into account also the number of features

# we optimize over all features (optimize score)

def custom_score(y_true, y_pred_probs, k):
    # selects top k samples with the highest posterior probability and sets them to 1
    # returns the ratio of samples correctly classified as 1 among them
    top_k_indices = np.argsort(y_pred_probs[:, 1])[::-1][:k]
    y_pred_k = np.zeros_like(y_true)
    y_pred_k[top_k_indices] = 1
    return precision_score(y_true, y_pred_k)

# 10 * (correct among 1000) - 200 * p
def ultimate_score(y_true, y_pred_probs, k, p):
    """
    Computes the ultimate score to optimize.
    The score is adapted to the total number of predicted 1s.
    It scales the ratio of correctly predicted 1s, so that the score corresponds to 1000 1s predictions.
    numpy.ndarray y_true: true labels
    numpy.ndarray y_pred_probs: probabilities of prediction
    int k: number of 1s to predict
    int p: number of features used for prediction
    :return: score to optimize
    """
    correct_fraction = custom_score(y_true, y_pred_probs, k)
    return 10 * np.floor(correct_fraction * 1000) - 200 * p
