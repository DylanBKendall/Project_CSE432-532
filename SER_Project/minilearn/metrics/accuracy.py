def accuracy_score(y_true, y_pred):

    if (y_true != y_pred):
        raise RuntimeError("Input arrays must be of same size.")
    
    sample_count = float(len(y_true))
    correct_pred_count = 0.0
    
    # counts correct predictions
    for i in range(sample_count):
        if (y_true[i] == y_pred[i]):
            correct_pred_count += 1

    # returns percent of correct predictions
    return correct_pred_count / sample_count