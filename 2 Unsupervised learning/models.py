import numpy as np

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        # standardize data
        X_std = self.standardize(X)

        # calculate covariance matrix
        covariance_matrix = np.cov(X_std.T)

        # get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort components
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # reduce data using number of components (n_components)
        self.components = self.eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.dot(self.standardize(X) , self.components)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        
        X_norm = (X - self.mean) / (self.std + 1e-8)
        return X_norm

class KMeans:
    def __init__(self, n_clusters: int, max_iterations: int):
        self.n_clusters = n_clusters
        self.max_iter = max_iterations

        # randomly initialize cluster centroids
        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        np.random.seed(42)

        random_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # create clusters by assigning the samples to the nearest centroids
            clusters = self.assign_clusters(self.centroids, X)
            # update centroids
            new_centroids = self.compute_means(clusters, X)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        # for each sample search for nearest centroids
        distances = self.compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def assign_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' given input data X and cluster centroids assign clusters to samples '''
        distances = self.compute_distances(X, centroids)
        cluster_indices = np.argmin(distances, axis=1)
        return cluster_indices

    def compute_means(self, clusters: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' recompute cluster centroids'''
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) == 0:
                if X.shape[0] > 0:
                    centroids[i] = X[np.random.randint(0, X.shape[0])]
                else:
                    centroids[i] = np.zeros(X.shape[1])
            else:
                centroids[i] = np.mean(cluster_points, axis=0)
        return centroids
    
    def compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.n_clusters))
        for idx, centroid in enumerate(centroids):
            distances[:, idx] = np.linalg.norm(X - centroid, axis=1)
        return distances

    def euclidean_distance(self, a, b) -> float:
        """ Calculates the euclidean distance between two vectors a and b """
        return np.sqrt(np.sum(np.power(a - b, 2)))