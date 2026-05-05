import numpy as np

def accuracy_score(y_true, y_pred):
    # convert to array
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (len(y_true) != len(y_pred)):
        raise RuntimeError("Input arrays must be of same size.")
    
    sample_count = len(y_true)
    correct_pred_count = 0.0
    
    # counts correct predictions
    for i in range(sample_count):
        if (y_true[i] == y_pred[i]):
            correct_pred_count += 1.0

    # returns percent of correct predictions
    return correct_pred_count / sample_count