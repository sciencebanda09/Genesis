"""
diagnostics.py — representation quality metrics for Genesis.

Tools to measure what the agent's latent representations actually encode.
No training loop hooks — these are called during evaluation/logging.

CKA (Centered Kernel Alignment, Kornblith et al. 2019): similarity between
    two representation matrices at different stages of processing. High CKA
    between encoder output and world-model state = the WM isn't discarding
    perceptual information. Low CKA between successive layers = information
    is progressively abstracted.

Linear probe: trains a quick logistic regression on frozen latents to predict
    ground-truth properties (position, velocity, object presence). If latents
    don't contain task-relevant info, the agent can't learn.

Latent collapse: measures whether representations collapse to a low-entropy
    subspace (all states map to the same latent). A collapsing network wastes
    capacity — this detects it before it hurts performance.
"""
import numpy as np


def hsic_kernel(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012). Lower bound of CKA."""
    m = len(X)
    K = X @ X.T
    L = Y @ Y.T
    np.fill_diagonal(K, 0.0); np.fill_diagonal(L, 0.0)
    K_mean = K.mean(axis=0, keepdims=True)
    L_mean = L.mean(axis=0, keepdims=True)
    K_c = K - K_mean - K_mean.T + K.mean()
    L_c = L - L_mean - L_mean.T + L.mean()
    return float((K_c * L_c).sum() / (m * (m - 3)))


def cka_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment between two representation matrices.

    Both (N, d1) and (N, d2). Returns a scalar in [0, 1].
    1.0 = representations are equivalent up to orthogonal transform.
    """
    m = len(X)
    K = X @ X.T
    L = Y @ Y.T
    H = np.eye(m) - np.ones((m, m)) / m
    K_c = H @ K @ H
    L_c = H @ L @ H
    div = np.sqrt((K_c * K_c).sum() * (L_c * L_c).sum()) + 1e-12
    return float((K_c * L_c).sum() / div)


def linear_probe(train_X, train_y, test_X, test_y, l2=1e-4):
    """Multinomial logistic regression as a linear probe.

    train_X, test_X: (N, D) float latents
    train_y, test_y: (N,) int labels
    Returns: test accuracy in [0, 1]
    """
    train_X = np.asarray(train_X, np.float32)
    test_X = np.asarray(test_X, np.float32)
    n_classes = int(train_y.max()) + 1
    D = train_X.shape[1]

    W = np.zeros((D, n_classes), np.float32)
    b = np.zeros(n_classes, np.float32)

    train_onehot = np.zeros((len(train_y), n_classes), np.float32)
    train_onehot[np.arange(len(train_y)), train_y] = 1.0

    lr = 0.01
    for _ in range(500):
        logits = train_X @ W + b
        e_x = np.exp(np.clip(logits, -30, 30) - logits.max(axis=-1, keepdims=True))
        probs = e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-9)
        d = (probs - train_onehot) / len(train_X)
        dW = train_X.T @ d + 2 * l2 * W
        db = d.sum(axis=0)
        W -= lr * dW
        b -= lr * db

    test_logits = test_X @ W + b
    preds = test_logits.argmax(axis=-1)
    return float((preds == test_y).mean())


def latent_collapse_metrics(latents: np.ndarray) -> dict:
    """Detect representation collapse.

    Args:
        latents: (N, D) batch of latent vectors from the encoder.
    Returns:
        dict with:
            fraction_variance_explained: cumulative fraction of total variance
                captured by the top principal components. Low values of the
                first component = no collapse.
            mean_cosine_sim: mean pairwise cosine similarity. High (>0.95)
                = near-collapse.
            effective_rank: number of components explaining 90% of variance.
                Close to 1 = collapse.
    """
    latents = np.asarray(latents, np.float32)
    mean = latents.mean(axis=0, keepdims=True)
    centered = latents - mean
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    var_frac = (s ** 2) / (s ** 2).sum()
    cumvar = np.cumsum(var_frac)
    eff_rank = int((cumvar < 0.90).sum()) + 1

    norms = np.linalg.norm(latents, axis=-1, keepdims=True) + 1e-8
    normalized = latents / norms
    cos_sim = normalized @ normalized.T
    tri = np.triu_indices(len(cos_sim), k=1)
    mean_cos = float(cos_sim[tri].mean()) if tri[0].size > 0 else 0.0

    return {
        "fraction_variance_explained": cumvar[:min(5, len(cumvar))].tolist(),
        "mean_cosine_sim": mean_cos,
        "effective_rank_90pct": eff_rank,
    }
