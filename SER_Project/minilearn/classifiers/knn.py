import numpy as np

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        neighbor_labels = self.y_train[neighbor_indices]

        values, counts = np.unique(neighbor_labels, return_counts=True)
        
        return values[np.argmax(counts)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))