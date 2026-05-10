import numpy as np

class LinearSVC:
    def __init__(self, C=1.0, learning_rate=0.001, max_iter=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights_ = None
        self.biases_ = None
        self.classes_ = None


    def _fit_binary(self, X, y_binary):
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(self.max_iter):
            margins = y_binary * (X @ w + b)
            misclassified = margins < 1

            grad_w = w - self.C * (y_binary[misclassified] @ X[misclassified])
            grad_b = -self.C * np.sum(y_binary[misclassified])

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        return w, b


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)

        self.weights_ = np.zeros((n_classes, X.shape[1]))
        self.biases_ = np.zeros(n_classes)

        for class_index, class_label in enumerate(self.classes_):
            y_binary = np.where(y == class_label, 1, -1)
            w, b = self._fit_binary(X, y_binary)
            self.weights_[class_index] = w
            self.biases_[class_index] = b

        return self


    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.weights_.T + self.biases_


    def predict(self, X):
        scores = self.decision_function(X)
        predicted_indices = np.argmax(scores, axis=1)

        return self.classes_[predicted_indices]