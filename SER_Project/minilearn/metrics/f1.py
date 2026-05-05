import precision
import accuracy

def f1_score(y_true, y_pred):

    precision_score = precision.precision_score(y_true, y_pred)
    accuracy_score = accuracy.accuracy_score(y_true, y_pred)

    # returns the harmonic mean of precision and accuracy scores
    return 2 / ((1.0 / precision_score) + (1.0 / accuracy_score))