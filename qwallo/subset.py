"""Pocket-local lining and CTQW-pair QUBO, shared by every protein."""

from itertools import combinations, islice

import numpy as np

from scipy.stats import rankdata

def rank_pocket_lining(residues, candidates, pockets, *, top_k=5, lining_mode="support"):
    """Rank the fixed pocket by apo cavity-wall support, without pair penalties.

    Use the same maximum-overlap family as the shared QUBO. Residue-number
    membership is only unambiguous for a single chain. Fail rather than silently
    treating absent geometry as evidence. Equal scores retain matrix-index order.
    Optional recurrence modes rank support*sqrt(frequency) (persistent) or
    support*sqrt(1-frequency) (variable), using the same matched apo family.
    """
    if lining_mode not in ("support", "persistent", "variable"):
        raise ValueError("unknown lining mode")
    ids = tuple(sorted(set(int(i) for i in candidates)))
    if (not 1 <= top_k <= len(ids)
            or any(i < 0 or i >= len(residues) for i in ids)):
        raise ValueError("valid pocket indices and subset size required")
    if len({residues[i].chain_id for i in ids}) != 1:
        raise ValueError("lining families require a single-chain pocket")
    numbers = {residues[i].residue_number for i in ids}
    families = tuple(pockets)
    if not families:
        raise ValueError("lining ranking requires cavity families")
    family = max(families, key=lambda p: len(numbers & set(p.residue_numbers))
                 / max(len(numbers | set(p.residue_numbers)), 1))
    support = dict(family.void_support)
    values = np.asarray([support.get(residues[i].residue_number, 0.0) for i in ids])
    if not np.isfinite(values).all() or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("positive finite cavity-wall support required")
    if lining_mode != "support":
        recurrence = dict(family.residue_persistence)
        if any(residues[i].residue_number not in recurrence for i, v in zip(ids, values) if v > 0):
            raise ValueError("recurrence required for every supported lining residue")
        frequency = np.asarray([recurrence.get(residues[i].residue_number, 0.0) for i in ids])
        if (not np.isfinite(frequency).all() or np.any(frequency < 0)
                or np.any(frequency > 1)):
            raise ValueError("lining recurrence must be finite and within [0, 1]")
        factor = frequency if lining_mode == "persistent" else 1.0 - frequency
        values = values * np.sqrt(factor)
        if not np.any(values > 0):
            raise ValueError("adjusted lining support must contain positive evidence")
    scores = np.full(len(residues), -1.0)
    scores[list(ids)] = rankdata(values, method="average") / len(ids)
    selected = tuple(ids[i] for i in np.argsort(-values, kind="stable")[:top_k])
    return selected, scores, ids

def build_lining_qubo(residues, candidates, connectivity, sources, pockets, *, top_k=5,
                      pair_weight=0.1, lining_mode="support"):
    """Lining unary plus bounded CTQW/distance redundancy, at fixed cardinality.

    The returned upper triangle represents sum(Q[i,j] x[i] x[j]); the
    solver enforces sum(x)=top_k. Sources are unused by this residue objective.
    Feature columns retain the shared export schema; only geometry is used.
    Geometry contains the selected lining-mode percentile. Positive pair_weight
    penalizes redundancy; negative weight rewards coupled nearby selections.
    """
    if not np.isfinite(pair_weight):
        raise ValueError("pair_weight must be finite")
    nodes = tuple(residues)
    _, scores, ids = rank_pocket_lining(
        nodes, candidates, pockets, top_k=top_k, lining_mode=lining_mode)
    C = np.asarray(connectivity, dtype=float)
    if (C.shape != (len(nodes), len(nodes)) or not np.isfinite(C).all()
            or np.any(C < 0) or not np.allclose(C, C.T)):
        raise ValueError("finite nonnegative symmetric CTQW map required")
    xyz = np.asarray([nodes[i].coordinate for i in ids], dtype=float)
    if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError("finite 3D residue coordinates required")
    utility = scores[list(ids)]
    Q = np.diag(-utility)
    first, second = np.triu_indices(len(ids), 1)
    if len(first) and top_k > 1:
        pair = C[np.asarray(ids)[first], np.asarray(ids)[second]]
        pair_rank = rankdata(pair) / len(pair)
        pair_rank[pair <= 0] = 0.0
        separation = np.linalg.norm(xyz[first] - xyz[second], axis=1)
        Q[first, second] = (pair_weight * np.minimum(utility[first], utility[second]) * pair_rank
                            * np.exp(-0.5 * (separation / 7.8)**2) / (top_k - 1))
    features = np.zeros((len(ids), 4))
    features[:, 2] = utility
    return Q, ids, features

def solve_pocket_qubo(qubo, *, top_k=5):
    """Enumerate fixed-size subsets in bounded-memory batches.

    Marginal[i] = max utility of a subset containing i. This supplies scores for
    all pocket residues instead of copying a single pocket score onto each node.
    Runtime is O(binomial(m,k)*k^2), not a quantum speedup.
    """
    Q = np.asarray(qubo, dtype=float)
    if (Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not np.isfinite(Q).all()
            or not np.allclose(np.tril(Q, -1), 0) or not 1 <= top_k <= len(Q)):
        raise ValueError("finite upper triangular Q and valid subset size required")
    iterator = combinations(range(len(Q)), top_k)
    diagonal = np.diag(Q)
    best = np.inf
    selected = None
    marginal = np.full(len(Q), -np.inf)
    while batch := list(islice(iterator, 8192)):
        groups = np.asarray(batch, dtype=int)
        energy = diagonal[groups].sum(axis=1)
        for i, j in combinations(range(top_k), 2):
            energy += Q[groups[:, i], groups[:, j]]
        index = int(np.argmin(energy))
        if energy[index] < best:
            best, selected = float(energy[index]), tuple(groups[index])
        np.maximum.at(marginal, groups.ravel(), np.repeat(-energy, top_k))
    return selected, best, marginal

def subset_scores(node_count, ids, selected, marginal):
    """Encode exact subset membership first, then optimal inclusion utility.

    The +1 offset resolves degenerate solutions in favor of the returned subset;
    report it explicitly because these are optimization scores, not occupancies.
    """
    values = np.asarray(marginal, dtype=float)
    scores = np.full(node_count, -1.0)
    scores[list(ids)] = (values - values.min()) / max(float(np.ptp(values)), 1e-15)
    scores[[ids[i] for i in selected]] += 1.0
    return scores
