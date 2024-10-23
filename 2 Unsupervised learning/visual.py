import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

dataset_path = "../dataset"

def visualize_embeddings(
    tsne_embeddings: np.ndarray, 
    pca_embeddings: np.ndarray, 
    labels: np.ndarray,
    name_file: str
):
    if tsne_embeddings.shape[1] == 2:
        tsne_df = pd.DataFrame(tsne_embeddings, columns=['tsne_x', 'tsne_y'])
        tsne_df['label'] = labels
        fig = px.scatter(tsne_df, x='tsne_x', y='tsne_y', color='label', title='t-SNE Embeddings (2D)')
        fig.write_image("results/" + name_file + "-tsne-2d.png")

    if tsne_embeddings.shape[1] == 3:
        tsne_df = pd.DataFrame(tsne_embeddings, columns=['tsne_x', 'tsne_y', 'tsne_z'])
        tsne_df['label'] = labels
        fig = px.scatter_3d(tsne_df, x='tsne_x', y='tsne_y', z='tsne_z', color='label', title='t-SNE Embeddings (3D)')
        fig.write_image("results/" + name_file + "-tsne-3d.png")
        fig.show()

    if pca_embeddings.shape[1] == 2:
        pca_df = pd.DataFrame(pca_embeddings, columns=['pca_x', 'pca_y'])
        pca_df['label'] = labels
        fig = px.scatter(pca_df, x='pca_x', y='pca_y', color='label', title='PCA Embeddings (2D)')
        fig.write_image("results/" + name_file + "-pca-2d.png")

    if pca_embeddings.shape[1] == 3:
        pca_df = pd.DataFrame(pca_embeddings, columns=['pca_x', 'pca_y', 'pca_z'])
        pca_df['label'] = labels
        fig = px.scatter_3d(pca_df, x='pca_x', y='pca_y', z='pca_z', color='label', title='PCA Embeddings (3D)')
        fig.write_image("results/" + name_file + "-pca-3d.png")
        fig.show()


def visualize_clusters(
    vImages_viz_orig: np.ndarray,
    cluster_labels_orig: np.ndarray,
    vImages_pca: np.ndarray,
    cluster_labels_pca: np.ndarray,
    labels: np.ndarray,
    name_file: str
):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("", "")
    )

    scatter_orig = go.Scatter3d(
        x=vImages_viz_orig[:, 0],
        y=vImages_viz_orig[:, 1],
        z=vImages_viz_orig[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=cluster_labels_orig.astype(float),
            colorscale='Viridis',
            opacity=0.8
        ),
        text=labels
    )
    fig.add_trace(scatter_orig, row=1, col=1)

    scatter_pca = go.Scatter3d(
        x=vImages_pca[:, 0],
        y=vImages_pca[:, 1],
        z=vImages_pca[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=cluster_labels_pca.astype(float),
            colorscale='Viridis',
            opacity=0.8
        ),
        text=labels
    )
    fig.add_trace(scatter_pca, row=1, col=2)

    fig.update_layout(
        title="",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        scene2=dict(
            xaxis_title='X ',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.write_image("results/" + name_file + ".png")


def visualize_nearest_images(texts, top_k_indices, images, num_samples=5):
    num_samples = min(num_samples, len(texts))
    random_indices = random.sample(range(len(texts)), num_samples)
    sampled_descriptions = [texts[i] for i in random_indices]
    sampled_top_k_indices = [top_k_indices[i] for i in random_indices]

    for i, text in enumerate(sampled_descriptions):
        indices = sampled_top_k_indices[i]
        visualize_nearest_images_single(text, indices, images)

def visualize_nearest_images_single(text, indices, images):
    _, axes = plt.subplots(1, len(indices) + 1, figsize=(15, 5))
    axes[0].set_title("Text")
    axes[0].text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
    axes[0].axis('off')
    for j, idx in enumerate(indices):
        image_path = f"{dataset_path}/flickr30k_images/{images[idx]}"
        img = Image.open(image_path)
        axes[j + 1].imshow(img)
        axes[j + 1].axis('off')
        axes[j + 1].set_title(f"Rank {j + 1}")

    plt.savefig("results/text" + text + ".png")
    plt.close()
