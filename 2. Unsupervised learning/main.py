import click
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import pandas as pd

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        # standardize data

        # calculate covariance matrix

        # get eigenvalues and eigenvectors

        # sort components

        # reduce data using number of components (n_components)

        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        # standardize data

        # reduce data using number of components
        pass

    def standardize(self, X: np.ndarray) -> np.ndarray:
        ''' Normalize data by substracting mean and divide by std '''
        X_norm = None
        return X_norm


class KMeans:
    def __init__(self, n_clusters: int, max_iterations: int):
        self.n_clusters = n_clusters
        self.max_iter = max_iterations

        # randomly initialize cluster centroids
        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        for _ in range(self.max_iter):
            # create clusters by assigning the samples to the nearest centroids
            clusters = self.assign_clusters(self.centroids, X)
            # update centroids
            new_centroids = self.compute_means(clusters, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # for each sample search for nearest centroids
        pass

    def assign_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' given input data X and cluster centroids assign clusters to samples '''
        pass

    def compute_means(self, clusters: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' recompute cluster centroids'''
        pass

    def euclidean_distance(self, a, b) -> float:
        """ Calculates the euclidean distance between two vectors a and b """
        return np.sqrt(np.sum(np.power(a - b, 2)))


def vectorize(images: np.ndarray) -> np.ndarray:
    pass


def neighbour_search(
        text_req: np.ndarray,
        vImages: np.ndarray,
        top_k: int = 5
) -> np.ndarray:
    ''' search for top_k nearest neightbours in the space '''
    pass


def load_data(image_folder: str, label_file: str) -> (np.array, np.array):
    data = pd.read_csv(f"{input_path}/labels.csv", delimiter='|')

@click.command()
@click.option('--input_path', type=str, help='Path to the input data')
@click.option('--n_components', type=int, help='Number of components')
@click.option('--n_clusters', type=str, help='Number of clusters')
def main(input_path, n_components, n_clusters):
    # load image data and text labels
    labels, descriptions, images = None

    # # vectorize images and text labels
    # vImages = vectorize(images)
    #
    # # PCA or t-SNE on images
    # # dimred = TSNE(n_components=n_components)
    # dimred = PCA(n_components=n_components)
    # dimred.fit(vImages)
    #
    # drvImages = dimred.transform(vImages)
    #
    # # Visualize 2D and 3D embeddings of images and color points based on labels
    # # TODO
    #
    # # Perform clustering on the embeddings and visualize the results
    # # clustere = AgglomerativeClustering(n_clusters=n_clusters)
    # clusterer = KMeans(n_clusters=n_clusters)
    #
    # # Visualize 2D and 3D embeddings of images and color points based on cluster label and original labels
    # # TODO
    #
    # # DBSCAN outlier detection
    # clusterer = DBSCAN(eps=None)  # select good eps value !!!
    #
    # # Create a copy of your trained data with cleaned outliers
    # # TODO
    #
    # # Select few text descriptions and select nearest neighbors based on embeddings.
    # vText = vectorize(descriptions)
    # drvText = dimred.transform(vText)
    # # TODO
    #
    # # Plot the results: text description, few nearest images
    # # TODO


if __name__ == "__main__":
    main()