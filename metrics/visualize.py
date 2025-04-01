import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns

from sklearn.manifold import TSNE




def visualize_tsne(data_tensor, labels_tensor, perplexity=30, learning_rate="auto", random_seed=0, n_iter=2000, title=""):
    
    
    data_np = np.array(data_tensor)
    labels_np = np.array(labels_tensor)

    print(data_np.shape)
    print(labels_np.shape)


    tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, random_state=random_seed, n_iter=n_iter)
    embedded_data = tsne.fit_transform(data_np)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels_np, cmap='viridis', s=15, alpha=0.6)
    plt.colorbar(scatter, boundaries=np.arange(len(np.unique(labels_np)) + 1) - 0.5).set_ticks(np.unique(labels_np))
    plt.title(title)
    plt.show()

    return embedded_data



def visualize_umap(data_tensors, labels_tensors, n_neighbors=50, min_dist=0.1, metric='cosine', title=''):

    data_np = np.array(data_tensors)
    labels_np = np.array(labels_tensors)

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,  # 2D
        metric=metric,
        n_epochs=500
    )
    u = fit.fit_transform(data_np)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(u[:, 0], u[:, 1], c=labels_np, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(scatter, boundaries=np.arange(len(np.unique(labels_np)) + 1) - 0.5).set_ticks(np.unique(labels_np))
    plt.title(title)
    plt.show()

def visualize_umap_nice(data_tensors, labels_tensors, n_neighbors=50, min_dist=0.1, metric='cosine', title='', save_path=None):
    data_np = np.array(data_tensors)
    labels_np = np.array(labels_tensors)

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,  # 2D
        metric=metric,
        n_epochs=500
    )
    u = fit.fit_transform(data_np)

    #sns.set(style='white', context='paper')
    palette = ['#DAE8FC', '#F8CECC']  # High contrast colors: Red and Dodger Blue
    edges = ['#6C8EBF', '#B85450']

    plt.figure(figsize=(5, 5))
    
    for label, marker, opacity, s in zip([0, 1], ['o', 'o'], [1.0, 0.4], [15, 5]):
        indices = labels_np == label
        plt.scatter(u[indices, 0], u[indices, 1], c=palette[label], alpha=opacity, edgecolor=edges[label], label=f'Class {label}', marker=marker, s=s, )
    
    # Customize the legend
    #plt.legend(loc='best', title='Classes', frameon=False)

    # Remove axes for a cleaner look
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Plot title
    if title:
        plt.title(title)

    # Layout adjustment
    plt.tight_layout()

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    # Show the plot
    plt.show()

def normalize_vectors(vectors):

    magnitudes = np.linalg.norm(vectors, axis=1)

    magnitudes[magnitudes == 0] = 1

    normalized_vectors = vectors / magnitudes[:, np.newaxis]
    return normalized_vectors

