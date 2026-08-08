import numpy as np


def standardize_matrix(matrix):
    """Standardize a matrix column-wise and return matrix, mean, std."""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    standardized = (matrix - mean) / std
    return standardized, mean, std


def fit_drs(clean_embeddings, num_directions=32, power=1.0, eps=1e-8):
    """
    Fit a DRS model on clean embeddings.

    DRS emphasizes shifts along low-variance directions in the clean data.
    """
    clean_embeddings = np.asarray(clean_embeddings, dtype=np.float32)
    if clean_embeddings.ndim != 2:
        raise ValueError("clean_embeddings must be a 2D array.")
    if len(clean_embeddings) < 2:
        raise ValueError("At least two clean embeddings are required for DRS.")

    x_std, mean, std = standardize_matrix(clean_embeddings)
    covariance = np.cov(x_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    max_dirs = min(num_directions, eigenvectors.shape[1])
    return {
        "mean": mean,
        "std": std,
        "eigenvalues": eigenvalues[:max_dirs],
        "eigenvectors": eigenvectors[:, :max_dirs],
        "power": power,
        "eps": eps,
    }


def drs_score(embedding, drs_model):
    """Compute DRS for a single embedding."""
    z = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    z_std = (z - drs_model["mean"]) / drs_model["std"]
    z_std = z_std[0]

    projections = np.abs(z_std @ drs_model["eigenvectors"]) ** drs_model["power"]
    denom = np.sqrt(np.maximum(drs_model["eigenvalues"], drs_model["eps"]))
    return float(np.sum(projections / denom))


def drs_threshold(clean_embeddings, quantile=0.99, num_directions=32, power=1.0):
    """Fit DRS and derive a threshold from clean samples."""
    model = fit_drs(
        clean_embeddings=clean_embeddings,
        num_directions=num_directions,
        power=power,
    )
    clean_scores = [drs_score(embedding, model) for embedding in clean_embeddings]
    threshold = float(np.quantile(clean_scores, quantile))
    return model, clean_scores, threshold
