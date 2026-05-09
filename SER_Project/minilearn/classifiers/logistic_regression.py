import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.classes = None
        self.weights = None
        self. bias = None

    def _softmax(self, logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)

        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), np.searchsorted(self.classes, y)] = 1

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for _ in range(self.max_iter):
            probs = self._softmax(X @ self.weights + self.bias)
            error = probs - one_hot

            self.weights -= self.learning_rate * (X.T @ error) / n_samples
            self.bias -= self.learning_rate * error.mean(axis=0)

    def predict_proba(self, X):
        return self._softmax(X @ self.weights + self.bias)

    def predict(self, X):
        return self.classes[self.predict_proba(X).argmax(axis=1)]