import re
import logging
import random
from tqdm import tqdm
import argparse    
import itertools
import time
from collections import defaultdict

import numpy as np
import faiss
import torch
import faiss
from openai import OpenAI
import openai
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

OPENAI_RETRIES = 3

# Define for local models so only one time initialization
MODEL = None
TOKENIZER = None
K_CANDIDATES = [3, 4, 5, 7, 10]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

RANDOM_SEED = 42
OPENAI_CLIENT = OpenAI(base_url="https://api.deepseek.com")

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hierarchical_cluster_fit_threshold(
    embeddings,
    thresholds,
    min_size_per_cluster=3,
    linkage="average"  # "ward" works poorly with cosine distance
):
    best_score = -1
    best_labels = None
    best_threshold = None

    for threshold in tqdm(thresholds):
        model = AgglomerativeClustering(
            n_clusters=None,
            linkage=linkage,
            metric="cosine",  # critical for normalized embeddings!
            distance_threshold=threshold
        )
        labels = model.fit_predict(embeddings)

        # Skip if any cluster is too small or only one cluster exists
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or np.any(counts < min_size_per_cluster):
            logging.info(f"Skipping threshold {threshold} due to cluster being too small")
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")
        logging.info(f"Threshold {threshold} has score of {score}")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_threshold = threshold

    # Fallback: If no threshold works, use a fixed k
    if best_threshold is None:
        feasible_k = max(2, len(embeddings) // min_size_per_cluster)
        model = AgglomerativeClustering(n_clusters=feasible_k, linkage=linkage, metric="cosine")
        best_labels = model.fit_predict(embeddings)
        best_threshold = "Fallback (k={})".format(feasible_k)

    return best_labels, best_threshold

def cluster_fit_k(embeddings, k_candidates, sample_size=20000):
    faiss.normalize_L2(embeddings)  # Critical for cosine similarity

    # Test k candidates
    scores = []

    for k in k_candidates:
        # FAISS K-means (GPU-accelerated)
        kmeans = faiss.Kmeans(
            d=embeddings.shape[1],
            k=k,
            niter=100,
        )
        kmeans.train(embeddings)
        logging.info(f"k={k}: Finished training")
        # Get cluster assignments and compute silhouette score
        _, labels = kmeans.index.search(embeddings, 1)
        labels = labels.flatten()
        score = silhouette_score(embeddings, labels, metric='cosine', sample_size=sample_size)
        scores.append(score)
        logging.info(f"k={k}: Silhouette Score = {score:.3f}")
    
    best_k = k_candidates[np.argmax(scores)]
    kmeans = faiss.Kmeans(
        d=embeddings.shape[1],
        k=best_k,
        niter=100,
    )
    kmeans.train(embeddings)
    return best_k, kmeans

def mmr_select(embeddings, selected, candidates, lambda_param=0.5, top_k=1):
    """Selects diverse points using Maximal Marginal Relevance (MMR)."""
    selected_set = embeddings[selected] if selected else np.zeros((1, embeddings.shape[1]))
    remaining = embeddings[candidates]
    sim_to_selected = cosine_similarity(remaining, selected_set).max(axis=1) if selected else np.zeros(len(candidates))
    sim_to_query = cosine_similarity(remaining, np.mean(selected_set, axis=0, keepdims=True)).flatten()
    
    mmr_scores = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
    mmr_indices = np.argsort(-mmr_scores)[:top_k]
    return [candidates[i] for i in mmr_indices]

def proportional_cluster_sample_with_diversity(kmeans, embeddings, total_samples, diversity_ratio=0.75):
    """Sample from clusters: 25% closest to centroid, 75% diverse using MMR."""
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()
    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
    logging.info(f"Unique clusters and cluster counts: {unique_clusters}; {cluster_counts}")
    n_clusters = len(unique_clusters)

    assert total_samples >= n_clusters, f"Need at least {n_clusters} samples (1/cluster)"

    proportions = cluster_counts / cluster_counts.sum()
    allocations = np.maximum(1, (proportions * (total_samples - n_clusters)).astype(int))
    allocations = allocations.cumsum()

    while allocations[-1] < total_samples:
        allocations[-1] += 1

    sampled_indices = []

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embs = embeddings[cluster_indices]
        centroid = kmeans.centroids[cluster]

        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        sorted_idx = np.argsort(dists)

        n_samples = allocations[i] - (0 if i == 0 else allocations[i - 1])
        n_close = max(1, int(n_samples * (1 - diversity_ratio)))
        n_diverse = n_samples - n_close

        # Local indices of the closest samples
        selected_close_local = sorted_idx[:n_close].tolist()

        # Local indices of the remaining candidates
        remaining_local = sorted_idx[n_close:].tolist()

        # MMR-based selection on local indices
        selected_diverse_local = []
        for _ in tqdm(range(n_diverse)):
            if not remaining_local:
                break
            selected = mmr_select(cluster_embs, selected_close_local + selected_diverse_local, remaining_local, lambda_param=0.7)
            selected_diverse_local.append(selected[0])
            remaining_local.remove(selected[0])

        # Map back to global indices
        final_indices = cluster_indices[selected_close_local + selected_diverse_local]
        sampled_indices.append(final_indices.tolist())

    return sampled_indices

def fast_cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def mmr_select_faiss(embeddings, selected, candidates, lambda_param=0.5, top_k=1):
    """Fast MMR selection using cosine similarity with FAISS candidates."""
    if not selected:
        selected_embs = np.mean(embeddings[candidates], axis=0, keepdims=True)
        sim_to_query = fast_cosine_similarity(embeddings[candidates], selected_embs).flatten()
        return [candidates[np.argmax(sim_to_query)]]

    selected_embs = embeddings[selected]
    candidate_embs = embeddings[candidates]

    sim_to_query = fast_cosine_similarity(candidate_embs, np.mean(selected_embs, axis=0, keepdims=True)).flatten()
    sim_to_selected = fast_cosine_similarity(candidate_embs, selected_embs).max(axis=1)

    mmr_scores = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
    top_indices = np.argsort(-mmr_scores)[:top_k]
    return [candidates[i] for i in top_indices]

def proportional_cluster_sample_with_diversity_faiss(kmeans, embeddings, total_samples, diversity_ratio=0.75, max_mmr_candidates=200):
    """Fast cluster sampling with centroid-faiss and approximate MMR."""
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()
    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_clusters)

    assert total_samples >= n_clusters, f"Need at least {n_clusters} samples (1 per cluster)."

    proportions = cluster_counts / cluster_counts.sum()
    allocations = np.maximum(1, (proportions * (total_samples - n_clusters)).astype(int)).cumsum()
    while allocations[-1] < total_samples:
        allocations[-1] += 1

    sampled_indices = []

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embs = embeddings[cluster_indices].astype(np.float32)
        centroid = kmeans.centroids[cluster].astype(np.float32).reshape(1, -1)

        # Use FAISS to get sorted indices by L2 distance to centroid
        index = faiss.IndexFlatL2(cluster_embs.shape[1])
        index.add(cluster_embs)
        _, I = index.search(centroid, len(cluster_embs))
        sorted_idx = I[0]

        n_samples = allocations[i] - (0 if i == 0 else allocations[i - 1])
        n_close = max(1, int(n_samples * (1 - diversity_ratio)))
        n_diverse = n_samples - n_close

        selected_close_local = sorted_idx[:n_close].tolist()
        remaining_local = sorted_idx[n_close:].tolist()

        # Subsample candidates to speed up MMR
        if len(remaining_local) > max_mmr_candidates:
            remaining_local = np.random.choice(remaining_local, size=max_mmr_candidates, replace=False).tolist()

        selected_diverse_local = []
        for _ in tqdm(range(n_diverse)):
            if not remaining_local:
                break
            selected = mmr_select_faiss(cluster_embs, selected_close_local + selected_diverse_local, remaining_local, lambda_param=0.7)
            selected_diverse_local.append(selected[0])
            remaining_local.remove(selected[0])

        final_indices = cluster_indices[selected_close_local + selected_diverse_local]
        sampled_indices.append(final_indices.tolist())

    return sampled_indices

def print_some_examples_each_group(sampled_indices, dataset):
    for i in range(len(sampled_indices)):
        logging.info(dataset[sampled_indices[i][0]])

def get_task_input(prompt):
    task_input = re.search(r'### TASK\s+(.*?)\s+### RESPONSE', prompt, re.DOTALL)
    return task_input

def request_openai_completion(message):
    for attempt in range(OPENAI_RETRIES):
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model="deepseek-chat",
                messages=message,
                max_tokens=8192,
            )

            return response.choices[0].message.content
        except openai.OpenAIError as e:
            if "rate" in str(e).lower():
                logging.warning("Hit rate limit; retrying...")
                time.sleep(61)
            else:
                logging.exception("Error calling OpenAI API:")
                time.sleep(1)
    logging.exception(f"Could not resolve error after {OPENAI_RETRIES} attempts")
    return None

def define_clusters(prompt_array, kmeans, embeddings_array, k_cluster, num_shots):
    # Get cluster assignments and distances
    distances, cluster_assignments = kmeans.index.search(embeddings_array, 1)
    cluster_assignments = cluster_assignments.flatten()
    clusters = defaultdict(list)
    for local_idx, (cluster_id, distance) in enumerate(zip(cluster_assignments, distances)):
        clusters[cluster_id].append((prompt_array[local_idx], distance[0]))
        
    # Get representative examples (closest to centroid)
    cluster_examples = {}
    for cluster_id in clusters:
        # Sort by distance ascending (smallest first)
        sorted_items = sorted(clusters[cluster_id], key=lambda x: x[1])
        cluster_examples[cluster_id] = [get_task_input(item[0]) for item in sorted_items[:num_shots]] # Get few-shot examples

    # Build cluster examples string
    clusters_str = "\n\n".join(
        f"Cluster {cid} examples:\n" + "\n".join(f"- {ex}" for ex in ex_list)
        for cid, ex_list in sorted(cluster_examples.items())
    )

    messages_cluster = [
        {"role": "user", "content": f"""
        {clusters_str}
        
        Analyze these {k_cluster} clusters of items and suggest a distinct category name for each without any explanation.
        The context behind such clustering is based on the topic.

        Rules:
        1. Use clear, specific names
        2. Maintain parallel structure
        3. No overlapping categories
        
        Output: Comma-separated names (one per cluster) without any explanations and maintain order starting from cluster 0 first.
        Example output format: Technology, Sports Apparel, Healthcare Devices
        
        Your categories:"""}
    ]

    clustered_categories = request_openai_completion(messages_cluster)
    categories = [cat.strip() for cat in clustered_categories.split(",")]

    # Based on the K-means result, assign based on the cluster id
    final_assignment = []
    for i, cluster_id in enumerate(cluster_assignments):
        final_assignment.append(categories[cluster_id])
    
    return final_assignment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General workflow for sampling')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help=f"Dataset id path.")
    parser.add_argument('--embedding_path', '-d', type=str,
                        help="Embedding path.")
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    set_seed()
    dataset = load_dataset(args.dataset_path)
    embeddings = np.load(args.embedding_path).astype('float32')
    prompts = dataset['prompt']

    # Clustering; Note that define_clusters can be omitted if not needed for sampling logic
    best_k, kmeans = cluster_fit_k(embeddings, K_CANDIDATES)
    cluster_assign = define_clusters(prompts, kmeans, embeddings, best_k, 10)

    # Sample indices relative to this source subset
    sampled_local_indices = proportional_cluster_sample_with_diversity_faiss(
        kmeans,
        embeddings,
        total_samples=200,
        diversity_ratio=0.75,
        max_mmr_candidates=20000
    )

    flat_sampled_indices = list(itertools.chain.from_iterable(sampled_local_indices))
    filtered_dataset = dataset.select(flat_sampled_indices)
