"""Classical and alternative-operator baselines for the same graph.

Every baseline consumes the identical adjacency matrix and the identical seed
set as the CTQW path, and returns a score vector on the same node order, so a
difference in performance cannot come from a different input.

* ``rwr_connectivity`` -- random walk with restart, the standard diffusion
  control. The restart probability is swept rather than fixed.
* ``coined_connectivity`` -- discrete-time coined quantum walk on the arc space
  with a Grover coin and a flip-flop shift, read out as a Cesaro time average.
  A coined walk is unitary, so an instantaneous distribution never converges;
  the time average is the discrete counterpart of the continuous-time limit.
"""

import numpy as np


def _checked(adjacency):
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(matrix) != 0.0):
        raise ValueError("adjacency must have a zero diagonal")
    if np.any(matrix.sum(axis=1) == 0.0):
        raise ValueError("adjacency contains isolated nodes")
    return matrix


def rwr_connectivity(adjacency, restart=0.15):
    """Return the exact stationary RWR matrix, column ``i`` = walk seeded at ``i``.

    Solved in closed form as ``restart * inv(I - (1 - restart) * P)`` with
    ``P`` the column-stochastic transition matrix, so there is no iteration
    tolerance to report. The diagonal is zeroed to match the CTQW convention.
    """
    matrix = _checked(adjacency)
    if not 0.0 < float(restart) < 1.0:
        raise ValueError("restart must lie in (0, 1)")
    transition = matrix / matrix.sum(axis=0, keepdims=True)
    size = len(matrix)
    stationary = float(restart) * np.linalg.solve(
        np.eye(size) - (1.0 - float(restart)) * transition, np.eye(size)
    )
    result = stationary.T
    np.fill_diagonal(result, 0.0)
    return np.clip(0.5 * (result + result.T), 0.0, None)


def coined_connectivity(adjacency, steps=1000, seeds=None):
    """Return Cesaro-averaged coined-walk occupancy for the requested seeds.

    Rows not in ``seeds`` stay zero: the arc-space walk costs O(steps * edges)
    per seed, so only the seeds a caller scores are propagated. Probability
    conservation is asserted at the final step.
    """
    matrix = _checked(adjacency)
    size = len(matrix)
    source, target = np.nonzero(matrix > 0)
    arc = {(int(u), int(v)): a for a, (u, v) in enumerate(zip(source, target))}
    degree = np.asarray([int((matrix[i] > 0).sum()) for i in range(size)])
    reverse = np.asarray([arc[(int(v), int(u))] for u, v in zip(source, target)])
    out_arcs = [np.flatnonzero(source == i) for i in range(size)]
    wanted = range(size) if seeds is None else tuple(dict.fromkeys(int(s) for s in seeds))

    result = np.zeros((size, size))
    for start in wanted:
        amplitude = np.zeros(len(source))
        amplitude[out_arcs[start]] = 1.0 / np.sqrt(degree[start])
        total = np.zeros(size)
        for _ in range(int(steps)):
            # Grover coin on each node's outgoing arcs, then flip-flop shift.
            node_sum = np.zeros(size)
            np.add.at(node_sum, source, amplitude)
            amplitude = 2.0 * (node_sum / degree)[source] - amplitude
            amplitude = amplitude[reverse]
            occupancy = np.zeros(size)
            np.add.at(occupancy, source, amplitude ** 2)
            total += occupancy
        total /= float(steps)
        if not np.isclose(total.sum(), 1.0, atol=1e-8):
            raise ValueError("coined walk lost probability mass")
        result[start] = total
    np.fill_diagonal(result, 0.0)
    return result
