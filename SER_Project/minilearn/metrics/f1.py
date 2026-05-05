import numpy as np

def get_recalls(y_true, y_pred):

    all_positive_count = {}
    true_positive_count = {}

    # counts the total number of positives and true positives for each class
    for true_class, predicted_class in zip(y_true, y_pred):
        all_positive_count[true_class] = all_positive_count.get(true_class, 0.0) + 1.0

        if predicted_class == true_class:
            true_positive_count[true_class] = true_positive_count.get(true_class, 0.0) + 1.0

    scores = []
    # calculates recall score for each class
    for key in all_positive_count.keys():
        scores.append(true_positive_count.get(key, 0) / all_positive_count[key])

    # returns class recall scores
    return scores

def get_precisions(y_true, y_pred):

    positive_count = {}
    true_positive_count = {}

    # counts the number of predicted positives and true positives for each class
    for true_class, predicted_class in zip(y_true, y_pred):
        positive_count[predicted_class] = positive_count.get(predicted_class, 0.0) + 1.0

        if predicted_class == true_class:
            true_positive_count[predicted_class] = true_positive_count.get(predicted_class, 0.0) + 1.0

    scores = []
    # calculates precision score for each class
    for key in positive_count.keys():
        scores.append(true_positive_count.get(key, 0) / positive_count[key])

    # returns class precision scores
    return scores

def f1_score(y_true, y_pred):
    # convert to array
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (len(y_true) != len(y_pred)):
        raise RuntimeError("Input arrays must be of same size.")
    
    precisions = get_precisions(y_true, y_pred)
    recalls = get_recalls(y_true, y_pred)

    f1_sum = 0.0

    # compute and add f1 score for each class
    for i in range(len(precisions)):
        f1_sum += 2 / ((1 / precisions[i]) + (1 / recalls[i]))

    # returns the average class f1 score
    return f1_sum / len(precisions)