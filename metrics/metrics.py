import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances


def entropy(input_array):
    _, counts = np.unique(input_array, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


def generate_sphere_points(dim, num_points, radius=1):
    points = np.random.randn(num_points, dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * radius


def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean()


def calculate_sample_alignment_distance(similarities, n_samples, labels):
    B = similarities.shape[0] // n_samples
    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]

    # Vectorize indices computation
    i_array = np.arange(n_samples).reshape(-1, 1)
    batch_offsets = np.arange(1, B) * n_samples
    indices_pos = i_array + batch_offsets  # Shape: (n_samples, B-1)

    # Compute distances without loops
    dist_to_positives = similarities[np.arange(n_samples)[:, None], indices_pos].mean(axis=1)

    sad_0 = dist_to_positives[indices_0]
    sad_1 = dist_to_positives[indices_1]

    return sad_0, sad_1


def calculate_sample_alignment_accuracy(sim, n_samples, labels, all_labels):
    B = sim.shape[0] // n_samples
    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]

    # Set diagonal to infinity to exclude self-matches
    np.fill_diagonal(sim, np.inf)
    index_top1_nn = np.argmin(sim, axis=1)

    # Determine if nearest neighbor is from the same sample across batches
    batch_indices = np.arange(sim.shape[0]) // n_samples
    sample_indices = np.arange(sim.shape[0]) % n_samples
    nn_batch_indices = index_top1_nn // n_samples
    nn_sample_indices = index_top1_nn % n_samples

    correct_match = (sample_indices == nn_sample_indices) & (batch_indices != nn_batch_indices)
    labels_batch = all_labels

    acc_cls0 = correct_match[indices_0].mean() * 100.0
    acc_cls1 = correct_match[indices_1].mean() * 100.0

    return acc_cls0, acc_cls1


def _compute_local_neighborhood_accuracies(sim, indices_0, indices_1, all_labels, r=0.05):
    med_n = int(sim.shape[0] * r)

    # Sort distances and get indices
    index_sorted = np.argsort(sim, axis=1)
    sorted_sim = np.take_along_axis(sim, index_sorted, axis=1)

    # Compute average distance for all samples
    avg_dist_med_all_lst = sorted_sim[:, :med_n].mean(axis=1)
    acc_med = all_labels[index_sorted[:, :med_n]]

    # For class 0
    sim_0 = sim[np.ix_(indices_0, indices_0)]
    index_sorted_0 = np.argsort(sim_0, axis=1)
    sorted_sim_0 = np.take_along_axis(sim_0, index_sorted_0, axis=1)
    avg_dist_med_0 = sorted_sim_0[:, :med_n].mean(axis=1)

    # For class 1
    sim_1 = sim[np.ix_(indices_1, indices_1)]
    index_sorted_1 = np.argsort(sim_1, axis=1)
    sorted_sim_1 = np.take_along_axis(sim_1, index_sorted_1, axis=1)
    avg_dist_med_1 = sorted_sim_1[:, :med_n].mean(axis=1)

    return avg_dist_med_0, avg_dist_med_1, acc_med


def calculate_class_alignment_distance(sim, all_embeddings, all_labels):
    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]
    sad_0, sad_1, _ = _compute_local_neighborhood_accuracies(sim, indices_0, indices_1, all_labels, r=1.0)
    return sad_0, sad_1


def calculate_class_alignment_consistency(sim, all_embeddings, all_labels):
    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]
    med_n = int(all_embeddings.shape[0] * 0.05)

    _, _, acc_med = _compute_local_neighborhood_accuracies(
        sim, indices_0, indices_1, all_labels, r=0.05)

    # Compute consistency for each class
    acc_med_0 = 1.0 - (acc_med[indices_0].sum(axis=1) / med_n)
    acc_med_1 = acc_med[indices_1].sum(axis=1) / med_n

    return acc_med_0 * 100.0, acc_med_1 * 100.0


def calculate_gaussian_potential_uniformity(all_embeddings, all_labels):
    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]

    # Convert to torch tensors
    embeddings_cls0 = torch.from_numpy(all_embeddings[indices_0])
    embeddings_cls1 = torch.from_numpy(all_embeddings[indices_1])

    gu_0 = lunif(embeddings_cls0)
    gu_1 = lunif(embeddings_cls1)

    return gu_0.item(), gu_1.item()


def calculate_probabilistic_entropy_uniformity(all_embeddings, all_labels, radius=1):
    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]

    uniformity_entropies_cls0 = []
    uniformity_entropies_cls1 = []

    for _ in range(5):
        points = generate_sphere_points(
            all_embeddings.shape[1], 10 * all_embeddings.shape[0], radius)

        sim_points = pairwise_distances(all_embeddings, points, metric="l2")
        closest_point = np.argmin(sim_points, axis=1)

        closest_point_min = closest_point[indices_0]
        closest_point_maj = closest_point[indices_1]

        entropy_cls0 = entropy(closest_point_min) / entropy(np.arange(closest_point_min.shape[0]))
        entropy_cls1 = entropy(closest_point_maj) / entropy(np.arange(closest_point_maj.shape[0]))

        uniformity_entropies_cls0.append(entropy_cls0)
        uniformity_entropies_cls1.append(entropy_cls1)

    return 100 * np.array(uniformity_entropies_cls0), 100 * np.array(uniformity_entropies_cls1)
