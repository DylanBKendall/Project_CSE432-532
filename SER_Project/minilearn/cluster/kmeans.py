import numpy as np
import random
import math

class KMeans:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.max_iter = 300
        self.tol = .0001
        self.L_ = int(math.log(n_clusters)) + 2
        self.cluster_centers_ = []
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None


    def _update_centroids(self, X):
        self.cluster_centers_.append(X[random.randint(0, X.shape[0] - 1)])

        for j in range(self.n_clusters - 1):
            distances = []

            for xi in range(X.shape[0]):
                smallest_dist = float('inf')

                for centroid in self.cluster_centers_:
                    curr_dist = 0

                    for i in range(X.shape[1]):
                        curr_dist += (centroid[i] - X[xi][i]) ** 2

                    smallest_dist = min(smallest_dist, curr_dist)

                distances.append(smallest_dist)

            distances = np.array(distances)

            best_candidate = None
            best_sum = float('inf')

            for x in range(self.L_):
                total = distances.sum()
                probs = distances / total

                candidate_idx = np.random.choice(X.shape[0], p=probs)
                candidate = X[candidate_idx]

                cand_dists = np.array([sum((candidate[i] - X[xi][i]) ** 2
                                           for i in range(X.shape[1]))
                                           for xi in range(X.shape[0])])
                
                new_distances = np.minimum(distances, cand_dists)
                curr_sum = new_distances.sum()

                if curr_sum < best_sum:
                    best_sum = curr_sum
                    best_candidate = candidate

            self.cluster_centers_.append(best_candidate)

        self.cluster_centers_ = np.array(self.cluster_centers_)
        return best_sum


    def fit(self, X, y=None, sample_weight=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        last_dist = self.update_centroids(X)

        difference = float('inf')
        n_iter = 0

        while difference > self.tol and n_iter < self.max_iter:
            labels = []

            for j in range(X.shape[0]):
                best_label, best_dist = 0, float('inf')

                for k, centroid in enumerate(self.cluster_centers_):

                    curr_dist = sum((centroid[i] - X[j][i]) ** 2 for i in range(X.shape[1]))

                    if curr_dist < best_dist:
                        best_dist, best_label = curr_dist, k

                labels.append(best_label)

            self.labels_ = np.array(labels)
            new_centers = []

            for k in range(self.n_clusters):
                pts = X[self.labels_ == k]
                new_centers.append(pts.mean(axis=0) if len(pts) > 0 else self.cluster_centers_[k])

            self.cluster_centers_ = np.array(new_centers)

            dist = sum(sum((self.cluster_centers_[self.labels_[j]][i] - X[j][i]) ** 2
                           for i in range(X.shape[1]))
                           for j in range(X.shape[0]))
            
            self.inertia_ = dist
            n_iter += 1
            difference = last_dist - dist
            last_dist = dist

        self.n_iter_ = n_iter
        return self


    def predict(self, X):
        X = np.asarray(X)
        labels = []

        for j in range(X.shape[0]):
            best_label, best_dist = 0, float('inf')

            for k, centroid in enumerate(self.cluster_centers_):
                curr_dist = sum((centroid[i] - X[j][i]) ** 2 for i in range(X.shape[1]))

                if curr_dist < best_dist:
                    best_dist, best_label = curr_dist, k

            labels.append(best_label)

        return np.array(labels)


    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X)