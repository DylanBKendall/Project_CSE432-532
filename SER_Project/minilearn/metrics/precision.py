import numpy as np

def precision_score(y_true, y_pred):
    # convert to array
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (len(y_true) != len(y_pred)):
        raise RuntimeError("Input arrays must be of same size.")
    
    positive_count = {}
    true_positive_count = {}

    # counts the number of predicted positives and true positives for each class
    for true_class, predicted_class in zip(y_true, y_pred):
        positive_count[predicted_class] = positive_count.get(predicted_class, 0.0) + 1.0

        if predicted_class == true_class:
            true_positive_count[predicted_class] = true_positive_count.get(predicted_class, 0.0) + 1.0

    scores = 0.0
    # sums precision scores of each class
    for key in true_positive_count.keys():
        scores += (true_positive_count[key] / positive_count[key])

    # returns average class precision score
    return scores / len(positive_count.keys())