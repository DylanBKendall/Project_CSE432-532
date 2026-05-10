import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def fit(self, features, labels):
        self.tree = self._build(features, labels)

    def predict(self, features):
        return np.array([self._traverse(sample, self.tree) for sample in features])

    def _gini(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        return 1 - np.sum((counts / len(labels)) ** 2)

    def _build(self, features, labels):
        if len(np.unique(labels)) == 1:
            return labels[0]

        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature_index in range(features.shape[1]):

            for threshold in np.unique(features[:, feature_index]):
                left = labels[features[:, feature_index] <= threshold]
                right = labels[features[:, feature_index] > threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(labels)
                
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature_index, threshold

        if best_feature is None:
            return Counter(labels).most_common(1)[0][0]

        mask = features[:, best_feature] <= best_threshold
        return {
            'feature': best_feature,
            'threshold': best_threshold,

            'left': self._build(features[mask], labels[mask]),
            'right': self._build(features[~mask], labels[~mask]),
        }

    def _traverse(self, sample, node):
        if not isinstance(node, dict):
            return node
        
        if sample[node['feature']] <= node['threshold']:
            return self._traverse(sample, node['left'])
        
        return self._traverse(sample, node['right'])