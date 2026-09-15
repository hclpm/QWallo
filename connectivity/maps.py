"""Exact classical evaluation of Laplacian continuous-time quantum walks."""

from typing import Iterable

import numpy as np

def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Return a numeric adjacency matrix after checking ENM graph invariants."""

    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("adjacency must be a square matrix with at least two nodes")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("adjacency must contain finite non-negative weights")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("adjacency must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12):
        raise ValueError("adjacency diagonal must be zero")
    if np.any(matrix.sum(axis=1) <= 0.0):
        raise ValueError("adjacency must not contain isolated nodes")
    return matrix

def time_averaged_ctqw_connectivity(
    adjacency: np.ndarray,
    times: Iterable[float] = np.geomspace(0.02, 20.0, 120),
    *,
    decay_time: float | None = None,
) -> np.ndarray:
    """Return time-averaged normalized-Laplacian CTQW probabilities.

    For each requested time, the transition amplitude is
    ``U(t) = exp(-i (L/lambda_max) t)``.  Entry ``[i, j]`` is the mean of
    ``|U(t)[j, i]|^2`` over time. If ``decay_time`` is provided, samples are
    weighted by ``exp(-t / decay_time)``. This routine uses exact
    eigendecomposition; the gate-model approximation lives in :mod:`quantum`.
    """

    matrix = _validate_adjacency(adjacency)
    time_array = np.asarray(tuple(times), dtype=float)
    if (
        time_array.ndim != 1
        or time_array.size == 0
        or not np.isfinite(time_array).all()
        or np.any(time_array <= 0.0)
    ):
        raise ValueError("times must be a non-empty sequence of positive finite values")
    if decay_time is not None and (
        not np.isfinite(decay_time) or decay_time <= 0.0
    ):
        raise ValueError("decay_time must be positive and finite")
    weights = (
        np.ones_like(time_array)
        if decay_time is None
        else np.exp(-time_array / decay_time)
    )

    # Dividing by lambda_max makes the dimensionless time grid less sensitive to
    # protein size and contact density.  It is a spectral, not a physical time unit.
    laplacian = np.diag(matrix.sum(axis=1)) - matrix
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues /= max(float(eigenvalues.max()), 1e-12)

    probabilities = np.zeros_like(matrix)
    for time, weight in zip(time_array, weights):
        # eigh gives L = V diag(lambda) V^T, so no dense matrix exponential is needed.
        amplitudes = (eigenvectors * np.exp(-1j * eigenvalues * time)) @ eigenvectors.T
        probabilities += weight * np.abs(amplitudes) ** 2
    probabilities /= weights.sum()
    # Self-return probability is not useful for ranking distal transmission.
    np.fill_diagonal(probabilities, 0.0)
    # Numerical roundoff can weakly break the expected symmetry of an undirected walk.
    return np.clip(0.5 * (probabilities + probabilities.T), 0.0, 1.0)
