import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigh
import time

def compute_dpp_matrix(item_vectors):
    """
    Compute the DPP kernel matrix L using cosine similarity.
    
    Args:
        item_vectors (np.ndarray): A (num_items, dimensions) matrix where each row is a item vector of certain dimensions.
    
    Returns:
        np.ndarray: The DPP kernel matrix L.
    """
    # Compute cosine similarity matrix
    L = cosine_similarity(item_vectors)

    # Ensure positive semi-definiteness by adding a small jitter term
    L += np.eye(L.shape[0]) * 1e-6
    
    return L


def squeeze_dpp_matrix(L, j_star, gamma=0.5):
    """
    Reduce the contribution of item item_id in the DPP matrix L.

    Args:
        L (np.ndarray): The DPP kernel matrix (num_items, num_items).
        item_id (int): The index of the item to squeeze.
        gamma (float): The weight of squeezing (0 < gamma < 1).

    Returns:
        np.ndarray: The modified DPP kernel matrix after squeezing.
    """
    # Step 1: Eigen-decomposition of L
    eigenvalues, V = eigh(L)  # L = V Λ V^T

    # Step 2: Find the most contributing eigenvector from this item (Equation 9)
    m = np.argmax(np.abs(V[j_star, :]))  # Index of max contribution

    # Step 3: Apply the squeezing formula (Equation 10)
    outer_product = np.outer(V[:, m], V[j_star, :] / V[j_star, m])
    V_prime = V - gamma * outer_product

    # Step 4: Reconstruct the squeezed kernel matrix
    L_squeezed = V_prime @ np.diag(eigenvalues) @ V_prime.T

    return L_squeezed


def k_dpp_sampling(L, k, epsilon=1e-8):
    """
    Perform k-DPP sampling from the kernel matrix L.

    Args:
        L (np.ndarray): The DPP kernel matrix (num_items, num_items).
        k (int): The number of items to sample.

    Returns:
        list: Indices of the selected items.
    """
    # Step 1: Eigen-decomposition of L
    eigenvalues, eigenvectors = eigh(L)

    # Step 2: Select k eigenvectors proportional to their eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative eigenvalues
    probs = eigenvalues / np.sum(eigenvalues)  # Normalise
    selected_indices = np.random.choice(len(eigenvalues), size=k, p=probs, replace=False)

    # Step 3: Select items via iterative volume maximization
    V = eigenvectors[:, selected_indices]  # Select the k chosen eigenvectors
    selected_items = []  # Indices of chosen items

    for _ in range(k):
        probs = np.sum(V**2, axis=1)  # Compute marginal gains
        probs /= np.sum(probs)  # Normalise probabilities
        chosen = np.random.choice(len(probs), p=probs)
        selected_items.append(chosen)

        # Update V using Gram-Schmidt orthogonalisation
        v_chosen = V[chosen, :].reshape(1, -1)  # Row vector
        V = V - (V @ v_chosen.T) @ v_chosen  # Project out component

        # Normalise while avoiding division by zero
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < epsilon] = 1  # Avoid division by zero
        V /= norms  # Normalise

    return list(set(selected_items))  # Ensure unique selections
    # return selected_indices


start_time = time.time()

# Compute DPP matrix
num_items = 500
dimensions = 64
item_vectors = np.random.rand(num_items, dimensions)
L = compute_dpp_matrix(item_vectors)    # L.shape is num_items x num_items

print("DPP Kernel Matrix Shape:", L.shape)

# ----------------
# k-DPP sampling
k = 5  # Number of items to sample
sampled_before = k_dpp_sampling(L, k)
print("Sampled item indices:", sampled_before)

# ----------------
# Squeeze matrix along an item
j_star = 4 
gamma = 0.8  # Squeezing weight

L_squeezed = squeeze_dpp_matrix(L, j_star, gamma)

sampled_after = k_dpp_sampling(L_squeezed, k)
print("Sampled item indices:", sampled_after)


similarity_before = np.mean(L[j_star, sampled_before])
similarity_after = np.mean(L[j_star, sampled_after])

print("Sampled before squeezing:", sampled_before, "Avg Sim Before:", similarity_before)
print("Sampled after squeezing:", sampled_after, "Avg Sim After:", similarity_after)

if similarity_after < similarity_before:
    print(f"Success: The new method reduced similarity (↓ {similarity_before - similarity_after:.4f})")
else:
    print("Warning: The squeezing did not reduce similarity as expected.")

end_time = time.time()
print("time taken:", (end_time-start_time))


exit()

def k_dpp_sampling(L, k):
    """Perform k-DPP sampling."""
    eigenvalues, eigenvectors = eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    probs = eigenvalues / np.sum(eigenvalues)
    selected_indices = np.random.choice(len(eigenvalues), size=k, p=probs, replace=False)
    return selected_indices

def orthogonalize_dpp_matrix(L, j_star, gamma=0.5):
    """Reduce the contribution of item j_star in the DPP matrix L by orthogonalization."""
    L = 0.5 * (L + L.T)  # Ensure symmetry

    # Extract the item vector
    v_j_star = L[j_star, :].reshape(-1, 1)  # Column vector

    # Normalize
    v_j_star = v_j_star / (np.linalg.norm(v_j_star) + 1e-8)

    # Compute the orthogonal projection matrix
    P = np.eye(L.shape[0]) - gamma * (v_j_star @ v_j_star.T)

    # Apply the projection
    L_squeezed = P @ L @ P

    # Ensure the matrix is still PSD
    eigenvalues, eigenvectors = eigh(L_squeezed)
    eigenvalues = np.maximum(eigenvalues, 0)  # Clip negative eigenvalues
    L_squeezed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return L_squeezed


L = np.random.rand(100, 100)
L = 0.5 * (L + L.T)  # Ensure symmetry
j_star = 3  # Item to squeeze
gamma = 0.3  # Strength of squeezing
k = 6  # Number of sampled items

sim_before = cosine_similarity(L)
sampled_before = k_dpp_sampling(L, k)
similarity_before = np.mean(sim_before[j_star, sampled_before])

L_squeezed = orthogonalize_dpp_matrix(L, j_star, gamma)

sim_after = cosine_similarity(L_squeezed)
sampled_after = k_dpp_sampling(L_squeezed, k)
similarity_after = np.mean(sim_after[j_star, sampled_after])

print("Sampled before squeezing:", sampled_before, "Avg Sim Before:", similarity_before)
print("Sampled after squeezing:", sampled_after, "Avg Sim After:", similarity_after)

if similarity_after < similarity_before:
    print(f"Success: The new method reduced similarity (↓ {similarity_before - similarity_after:.4f})")
else:
    print("Warning: The squeezing did not reduce similarity as expected.")