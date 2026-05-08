import math
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        sample_count = X.shape[0]
        feature_count = X.shape[1]

        self.weights = np.zeros(feature_count)
        self.bias = 0

        for i in range(self.max_iter):
            predictions = self.sigmoid(X @ self.weights + self.bias)
            error = predictions - y
            
            # @ is vector multiplication
            self.weights -= self.learning_rate * (X.T @ error) / sample_count
            self.bias -= self.learning_rate * np.sum(error) / sample_count

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        return self.sigmoid(X @ self.weights + self.bias)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))