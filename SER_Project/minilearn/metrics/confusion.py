import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (len(y_true) != len(y_pred)):
        raise RuntimeError("Input arrays must be of same size.")

    classes = list(set(y_true))
    
    confusion = pd.DataFrame(0, index=[classes], columns=[classes])
    confusion = confusion.rename_axis(index="true", columns="pred")

    for true_class, predicted_class in zip(y_true, y_pred):
        confusion.loc[true_class, predicted_class] += 1
    
    return confusion