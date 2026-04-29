import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Prevent divide-by-zero for constant columns
        self.scale_[self.scale_ == 0] = 1.0

        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit().")

        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)