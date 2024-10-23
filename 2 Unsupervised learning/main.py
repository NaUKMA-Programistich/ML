import click
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from vectorize import vectorize_images, vectorize_text
from models import PCA, KMeans
from visual import visualize_embeddings, visualize_clusters, visualize_nearest_images

def load_data() -> tuple[(np.array, np.array, np.array)]:
    data = pd.read_csv(f"../dataset/labels.csv", delimiter='|', skipinitialspace=True)

    images = data['image_name'].tolist()
    labels = data['label'].tolist()
    descriptions = data['comment'].tolist()

    return labels, descriptions, images

def validation_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    stratify: bool = False
) -> tuple[(np.ndarray, np.ndarray, np.ndarray, np.ndarray)]:
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_param
    )

    return X_train, X_test, y_train, y_test

def neighbour_search(text_req: np.ndarray, vImages: np.ndarray, top_k: int = 5) -> np.ndarray:
    similarities = np.dot(text_req, vImages.T)
    return np.argsort(-similarities, axis=1)[:, :top_k]

@click.command()
@click.option('--input_path', type=str, help='Path to the input data')
@click.option('--n_components', type=int, help='Number of components')
@click.option('--n_clusters', type=str, help='Number of clusters')
def main(input_path, n_components, n_clusters):
    main_internal(n_components, n_clusters)


def main_internal(n_components, n_clusters):
    # load image data and text labels
    labels, descriptions, images = load_data()
    print(f"Loaded {len(images)} images and {len(descriptions)} descriptions")

    # vectorize images and text labels
    vImages = vectorize_images(images)
    print(f"Vectorized images to shape {vImages.shape}")

     # PCA or t-SNE on images
    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_2d_embeddings = tsne_2d.fit_transform(vImages)
    tsne_3d = TSNE(n_components=3, random_state=42)
    tsne_3d_embeddings = tsne_3d.fit_transform(vImages)
    print(f"t-SNE embeddings shapes: {tsne_2d_embeddings.shape}, {tsne_3d_embeddings.shape}")

    pca_2d = PCA(n_components=2)
    pca_2d.fit(vImages)
    pca_2d_embeddings = pca_2d.transform(vImages)
    pca_3d = PCA(n_components=3)
    pca_3d.fit(vImages)
    pca_3d_embeddings = pca_3d.transform(vImages)
    print(f"PCA embeddings shapes: {pca_2d_embeddings.shape}, {pca_3d_embeddings.shape}")

    # Visualize 2D and 3D embeddings of images and color points based on labels
    visualize_embeddings(tsne_2d_embeddings, pca_2d_embeddings, labels, name_file="2d")
    visualize_embeddings(tsne_3d_embeddings, pca_3d_embeddings, labels, name_file="3d")

    # Perform clustering on the embeddings and visualize the results
    kmeans_org = KMeans(n_clusters=n_clusters, max_iterations=100)
    kmeans_org.fit(vImages)
    cluster_labels_org = kmeans_org.predict(vImages)

    kmeans_pca = KMeans(n_clusters=n_clusters, max_iterations=100)
    kmeans_pca.fit(pca_3d_embeddings)
    cluster_labels_pca = kmeans_pca.predict(pca_3d_embeddings)

    # Visualize 3D embeddings of images and color points based on cluster label and original labels
    visualize_clusters(vImages[:, :3], cluster_labels_org, pca_3d_embeddings, cluster_labels_pca, labels, name_file="KMeans")

    silhouette_scores = []
    cluster_range = range(2, 10)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, max_iterations=100)
        kmeans.fit(pca_3d_embeddings)
        labels_k = kmeans.predict(pca_3d_embeddings)
        score = silhouette_score(pca_3d_embeddings, labels_k)
        silhouette_scores.append(score)
        print(f"Silhouette score for k={k}: {score}")
    
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels_hierarchical = agglo.fit_predict(pca_3d_embeddings)
    visualize_clusters(pca_3d_embeddings, cluster_labels_pca, pca_3d_embeddings, cluster_labels_hierarchical, labels, name_file="AgglomerativeClustering")

    # DBSCAN outlier detection
    X_train, _, y_train, _ = validation_split(pca_3d_embeddings, labels)
    dbscan = DBSCAN(eps=0.7, min_samples=3)
    dbscan_labels = dbscan.fit_predict(X_train)
    print(f"DBSCAN labels shape: {dbscan_labels.shape}")
    visualize_clusters(X_train, kmeans_pca.predict(X_train), X_train, dbscan_labels, y_train, name_file="DBSCAN")

    # Create a copy of your trained data with cleaned outliers 
    non_outlier_indices = np.where(dbscan_labels != -1)[0]
    X_train_clean = X_train[non_outlier_indices]

    kmeans_clean = KMeans(n_clusters=n_clusters, max_iterations=100)
    kmeans_clean.fit(X_train_clean)
    kmeans_clean.predict(X_train_clean)
    print(f"Number of samples before cleaning: {X_train.shape[0]}, after cleaning: {X_train_clean.shape[0]}")

    # Select few text descriptions and select nearest neighbors based on embeddings. 
    vTexts = vectorize_text(descriptions)
    pca_3d.transform(vTexts)
    
    # Plot the results: text description, few nearest images
    top_k_indices = neighbour_search(vTexts, vImages, top_k=5)
    for i, text in enumerate(descriptions):
        print(f"Text query: {text}")
        indices = top_k_indices[i]
        for rank, idx in enumerate(indices):
            image_name = images[idx]
            description = descriptions[idx]
            print(f"Rank {rank+1}: Image {image_name}, Description: {description}")
        print("\n")
    
    visualize_nearest_images(descriptions, top_k_indices, images)

if __name__ == "__main__":
    main_internal(n_components=3, n_clusters=6)