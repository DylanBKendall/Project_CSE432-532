import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = []
        self.scale_ = []

    # sets mean and std to that of x
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        # calc mean and std
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X, y=None):

        if (not np.any(self.mean_)):
            raise RuntimeError("Scaler must first be fit.")

        X = np.asarray(X, dtype=float)

        # returns data scaled to a mean of 0
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)