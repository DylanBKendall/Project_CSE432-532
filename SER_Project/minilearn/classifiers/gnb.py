import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for class_label in self.classes:
            features_in_class = X[y == class_label]

            self.class_means[class_label] = features_in_class.mean(axis=0)
            self.class_variances[class_label] = features_in_class.var(axis=0)
            self.class_priors[class_label] = len(features_in_class) / len(X)

    def _log_likelihood(self, sample, class_label):
        mean = self.class_means[class_label]
        variance = self.class_variances[class_label]
        
        return np.sum(np.log(2 * np.pi * variance) + (sample - mean) ** 2 / variance) * -.5

    def predict(self, X):
        predictions = []

        for sample in X:
            class_scores = {
                class_label: np.log(self.class_priors[class_label]) + self._log_likelihood(sample, class_label)
                for class_label in self.classes
            }
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)