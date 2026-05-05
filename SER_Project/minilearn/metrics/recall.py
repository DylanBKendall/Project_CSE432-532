def recall_score(y_true, y_pred):

    if (len(y_true) != len(y_pred)):
        raise RuntimeError("Input arrays must be of same size.")
    
    all_positive_count = 0.0
    true_positive_count = 0.0

    # counts the total number of positives and true positives
    for i in range(len(y_true)):
        if (y_true[i] == True):
            all_positive_count += 1
            if (y_pred[i] == True):
                true_positive_count += 1

    # returns percent of true positives in positive predictions
    return true_positive_count / all_positive_count