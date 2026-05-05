def accuracy_score(y_true, y_pred):
    
    sample_count = float(len(y_true))
    correct_pred_count = 0.0
    for i in range(sample_count):
        if (y_true[i] == y_pred[i]):
            correct_pred_count += 1
    return correct_pred_count / sample_count