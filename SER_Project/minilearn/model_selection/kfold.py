import math
import numpy as np

class KFold:
    def __init__(self, n_splits = 5):
        self.n_splits = n_splits

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):

        samples = X.shape[0]
        fold_sizes = np.full(self.n_splits, samples // self.n_splits)
        fold_sizes[:samples % self.n_splits] += 1

        curr = 0

        for fold_size in fold_sizes:
            next_fold = curr + fold_size

            train = np.concatenate((np.arange(0, curr), np.arange(next_fold, samples)))
            test = np.arange(curr, next_fold)
            
            yield train, test

            curr = next_fold