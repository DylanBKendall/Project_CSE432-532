import math
import numpy as np

class LogisticRegression:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        feature_count = len(X)
        sample_count = X.shape()[0]

        self.weights = np.zeros(feature_count)
        self.bias = 0

        for i in range(self.max_iter):
            


    def predict(self, X):
        

    def predict_proba(self, X):


    def score(self, X, y):
