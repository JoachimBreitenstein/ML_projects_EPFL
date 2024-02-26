import numpy as np
from scipy.stats import norm
from numpy.linalg import inv
from typing import List
from itertools import combinations


def CItest(D: np.ndarray, X: int, Y: int, Z: List[int]) -> int:
    """
    Test for conditional independence between variables X and Y given a set Z in dataset D.

    Parameters:
    D (numpy.ndarray): A matrix of data with size n*p, where n is the number of samples and p is the number of variables.
    X (int): Index of the first variable (zero-based indexing).
    Y (int): Index of the second variable (zero-based indexing).
    Z (List[int]): A list of indices for variables in the conditioning set (zero-based indexing).

    Returns:
    int: 1 if conditionally independent, 0 otherwise.
    """
    
    # Significance level
    alpha = 0.06

    # Number of samples
    n = D.shape[0]

    # Select columns corresponding to X, Y, and Z from the dataset
    DD = D[:, [X, Y] + Z]
    
    # Compute the precision matrix
    R = np.corrcoef(DD, rowvar=False)
    P = inv(R)

    # Calculate the partial correlation coefficient and Fisher Z-transform
    ro = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    zro = 0.5 * np.log((1 + ro) / (1 - ro))

    # Test for conditional independence
    c = norm.ppf(1 - alpha / 2)
    if abs(zro) < c / np.sqrt(n - len(Z) - 3):
        CI = 1
    else:
        CI = 0

    return CI


def sgs_algorithm(data):
    """
    Implement the SGS (Spirtes-Glymour-Scheines) algorithm to learn the skeleton
    of the underlying DAG from the data.

    Args:
    data (np.ndarray): The data matrix (n_samples, n_features).

    Returns:
    np.ndarray: The adjacency matrix of the learned skeleton.
    """
    n_variables = data.shape[1]
    skeleton = np.ones((n_variables, n_variables), dtype=int) - np.eye(n_variables, dtype=int)

    for x, y in combinations(range(n_variables), 2):
        # Check conditional independence for all subsets Z of the remaining variables
        for z_size in range(n_variables - 2):
            for z in combinations(set(range(n_variables)) - {x, y}, z_size):
                if CItest(data, x, y, list(z)):
                    skeleton[x, y] = skeleton[y, x] = 0  # Remove edge X-Y from the skeleton
                    break  # Break out of the loop as we found a set Z that makes X and Y conditionally independent

    return skeleton

# Assume we have a data matrix D, let's simulate it for the purpose of this example
# Normally, we would load the data from the given datasets (D1, D2, D3)
n_samples = 500
n_variables = 5
D = np.random.randn(n_samples, n_variables)  # Simulated data matrix
for i in range(3):
    D = np.load(f"Q6_files/D{i+1}.npy")
    # Apply the SGS algorithm to learn the skeleton
    skeleton_adjacency_matrix = sgs_algorithm(D)
    print(f"Skeleton ajencency matrix for datafile D{i+1}:", skeleton_adjacency_matrix)