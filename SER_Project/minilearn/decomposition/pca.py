import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components_ = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, data):
        self.mean_ = np.mean(data, axis=0)
        centered_data = data - self.mean_

        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components_ = eigenvectors[:, :self.n_components_].T
        self.explained_variance_ratio_ = eigenvalues[:self.n_components_] / np.sum(eigenvalues)
        
        return self

    def transform(self, data):
        return (data - self.mean_) @ self.components_

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, projected_data):
        return projected_data @ self.components_.T + self.mean_