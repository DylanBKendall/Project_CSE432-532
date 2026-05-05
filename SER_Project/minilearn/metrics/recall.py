import numpy as np

def recall_score(y_true, y_pred):
    # convert to array
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (len(y_true) != len(y_pred)):
        raise RuntimeError("Input arrays must be of same size.")
    
    all_positive_count = {}
    true_positive_count = {}

    # counts the total number of positives and true positives for each class
    for true_class, predicted_class in zip(y_true, y_pred):
        all_positive_count[true_class] = all_positive_count.get(true_class, 0.0) + 1.0

        if predicted_class == true_class:
            true_positive_count[true_class] = true_positive_count.get(true_class, 0.0) + 1.0

    scores = 0.0
    # sums recall scores of each class
    for key in true_positive_count.keys():
        scores += (true_positive_count[key] / all_positive_count[key])

    # returns average class recall score
    return scores / len(all_positive_count.keys())