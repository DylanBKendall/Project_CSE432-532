import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, min_samples_leaf=1, max_depth=None):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tree_ = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / len(y)) ** 2)
    

    def _traverse(self, sample, node):
        if not isinstance(node, dict):
            return node
        
        if sample[node['feature']] <= node['threshold']:
            return self._traverse(sample, node['left'])
        
        return self._traverse(sample, node['right'])


    def _build(self, X, y, depth=0):
        if (len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth == self.max_depth):
            return Counter(y).most_common(1)[0][0]

        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature_index in range(X.shape[1]):

            for threshold in np.unique(X[:, feature_index]):
                left = y[X[:, feature_index] <= threshold]
                right = y[X[:, feature_index] > threshold]

                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature_index, threshold

        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        mask = X[:, best_feature] <= best_threshold
        return {
            'feature': best_feature,
            'threshold': best_threshold,

            'left': self._build(X[mask], y[mask]),
            'right': self._build(X[~mask], y[~mask]),
        }


    def fit(self, X, y):
        self.tree_ = self._build(np.array(X), y)


    def predict(self, X):
        return np.array([self._traverse(sample, self.tree_) for sample in np.array(X)])