import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    # sets mean and std to that of x
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        # calc mean and std
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X):
        if (self.mean_ == None or self.scale_ == None):
            raise RuntimeError("StandardScaler must be fit().")
        if (self.scale_ == 0):
            return X

        X = np.asarray(X, dtype=float)

        # returns data scaled to a mean of 0
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)