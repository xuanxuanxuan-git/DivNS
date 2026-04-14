import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import cosine_similarity

def compute_dpp_matrix(item_vectors):
    """Compute the DPP similarity matrix using cosine similarity."""
    return cosine_similarity(item_vectors)

def k_dpp_sampling(L, k):
    """Perform k-DPP sampling."""
    eigenvalues, eigenvectors = eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative values
    probs = eigenvalues / np.sum(eigenvalues)  # Sampling probability
    selected_indices = np.random.choice(len(eigenvalues), size=k, p=probs, replace=False)
    return selected_indices

def project_away_from_item(item_vectors, j_star, gamma=0.5):
    """
    Modify item vectors to push them away from the influence of item j_star.
    """
    v_j_star = item_vectors[j_star].reshape(1, -1)
    norm_v_j_star = v_j_star / (np.linalg.norm(v_j_star) + 1e-8)

    for i in range(item_vectors.shape[0]):
        v_i = item_vectors[i].reshape(1, -1)
        projection = (np.dot(v_i, norm_v_j_star.T) * norm_v_j_star).reshape(-1)
        item_vectors[i] -= gamma * projection  # Push away

    return item_vectors

def compute_avg_similarity(item_vectors, sampled_items, item_id):
    """Compute average cosine similarity between a target item and sampled items."""
    similarities = [
        cosine_similarity(item_vectors[item_id].reshape(1, -1), item_vectors[i].reshape(1, -1))[0, 0]
        for i in sampled_items
    ]
    return np.mean(similarities)


num_items = 500
dimensions = 64
item_vectors = np.random.rand(num_items, dimensions)

# Compute original DPP matrix
L = compute_dpp_matrix(item_vectors)

# Select an item to suppress
j_star = 31  
gamma = 0.3  # suppression degree
k = 10  # Number of items to sample

# Compute similarity before squeezing
sampled_before = k_dpp_sampling(L, k)
similarity_before = compute_avg_similarity(item_vectors, sampled_before, j_star)

# Modify item embeddings to push them away from j_star
modified_vectors = project_away_from_item(item_vectors.copy(), j_star, gamma)

# Compute new L after modifying vectors
L_squeezed = compute_dpp_matrix(modified_vectors)

# Compute similarity after squeezing
sampled_after = k_dpp_sampling(L_squeezed, k)
similarity_after = compute_avg_similarity(modified_vectors, sampled_after, j_star)

print("Sampled before squeezing:", sampled_before, "Avg Sim Before:", similarity_before)
print("Sampled after squeezing:", sampled_after, "Avg Sim After:", similarity_after)

if similarity_after < similarity_before:
    print(f"✅ Success: The method reduced similarity (↓ {similarity_before - similarity_after:.4f})")
else:
    print("⚠️ Warning: The squeezing did not reduce similarity as expected.")