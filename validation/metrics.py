"""Three-tier evaluation of a residue score against the allosteric site.

The tiers escalate in how hard they are to pass:

1. ``random background`` -- positives against every non-seed residue. The floor
   is the positive prevalence. An allosteric site is buried and far from the
   active site by definition, so a score that measures only those two
   properties already clears this tier; passing it proves little.
2. ``condition-matched fake pockets`` -- positives against pooled residues of
   control pockets whose centres match the true site in burial and in graph
   distance to the seed. Two separate hard gates, never summed, so a good
   burial match cannot pay for a bad distance match.
3. ``hits at k`` -- how many of the submitted k residues are ground truth. This
   is the challenge output format and the strictest read.

``lift`` = AUPRC / floor is the comparable quantity across targets, because the
floor differs with prevalence and with control count. Four control scores run
through the identical path; if ``burial`` or ``neg_seed_dist`` lifts above 1.5
the matching gate admitted a skewed control set and the result of interest is
not interpretable.
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata


def average_precision(labels, scores):
    """Tie-aware average precision; ties share one precision/recall point."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("average precision requires positive and negative labels")
    order = np.argsort(-scores, kind="stable")
    ordered_labels, ordered_scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(labels) - 1]
    true_positives = np.cumsum(ordered_labels)[ends]
    return float(np.sum(
        np.diff(np.r_[0, true_positives]) / positives * true_positives / (ends + 1)
    ))


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("auroc requires positive and negative labels")
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def burial(coordinates, radius=10.0):
    """Neighbour count within ``radius``; the C-alpha coordination proxy."""
    xyz = np.asarray(coordinates, dtype=float)
    distances = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    return ((distances <= radius).sum(axis=1) - 1).astype(float)


def seed_hops(adjacency, seed_indices):
    """Minimum hop count from any seed, on the unweighted contact graph."""
    hops = shortest_path(np.asarray(adjacency, dtype=float),
                         directed=False, unweighted=True)
    return hops[list(seed_indices)].min(axis=0)


def fake_pockets(coordinates, adjacency, seed_indices, positive_indices, *,
                 size=5, radius=8.0, overlap_max=0.4, rounds=20,
                 match_tol=0.10, min_negatives=20):
    """Build control pockets matched to the true site in burial and seed distance.

    Matching properties are measured on the whole positive set. A centre is
    eligible when it is neither seed nor positive and its own burial and seed
    distance both fall inside the relative tolerance. A pocket is the centre
    plus its nearest eligible residues within ``radius``; it is accepted only if
    at most ``overlap_max * size`` of its members are already used by an earlier
    accepted pocket, in this round or any previous one. The tolerance escalates
    to 0.30 in 0.05 steps until ``min_negatives`` pooled residues are reached.

    Returns ``(control_indices, report)``. ``control_indices`` is empty when the
    target is not evaluable at any tolerance.
    """
    xyz = np.asarray(coordinates, dtype=float)
    bur = burial(xyz)
    hop = seed_hops(adjacency, seed_indices)
    positives = np.asarray(sorted(set(int(i) for i in positive_indices)), dtype=int)
    seeds = set(int(i) for i in seed_indices)
    if not len(positives):
        raise ValueError("fake pockets need at least one positive residue")
    target_bur = float(bur[positives].mean())
    target_hop = float(hop[positives].mean())
    distances = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    forbidden = seeds | set(positives.tolist())

    tolerance = float(match_tol)
    while True:
        eligible = np.asarray([
            i for i in range(len(xyz))
            if i not in forbidden
            and abs(bur[i] - target_bur) <= tolerance * max(target_bur, 1.0)
            and abs(hop[i] - target_hop) <= tolerance * max(target_hop, 1.0)
        ], dtype=int)
        eligible_set = set(int(i) for i in eligible)
        used, accepted, per_round = set(), [], []
        for index in range(int(rounds)):
            generator = np.random.default_rng(index)
            pool = [int(i) for i in eligible]
            generator.shuffle(pool)
            count = 0
            for centre in pool:
                near = [
                    int(j) for j in np.argsort(distances[centre])
                    if int(j) != centre and int(j) in eligible_set
                    and distances[centre, j] <= radius
                ][: size - 1]
                if len(near) < size - 1:
                    continue
                members = set([centre] + near)
                if len(members & used) > overlap_max * size:
                    continue
                used |= members
                accepted.append(sorted(members))
                count += 1
            per_round.append(count)
        control = np.asarray(sorted(used - forbidden), dtype=int)
        if len(control) >= min_negatives or tolerance >= 0.30 - 1e-9:
            break
        tolerance = round(tolerance + 0.05, 2)

    report = dict(
        n_negatives=int(len(control)), n_pockets=int(len(accepted)),
        match_tol_used=float(tolerance), per_round_pockets=per_round,
        n_eligible_centres=int(len(eligible)),
        target_burial=target_bur, target_seed_hop=target_hop,
        smd_burial=_smd(bur[positives], bur[control]) if len(control) else None,
        smd_seed_hop=_smd(hop[positives], hop[control]) if len(control) else None,
        evaluable=bool(len(control) >= min_negatives),
    )
    return control, report


def _smd(a, b):
    """Standardised mean difference; a balance diagnostic, never a filter."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt(0.5 * (a.var() + b.var()))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def scope_metrics(scores, positive_indices, candidate_indices):
    """AUPRC with its floor and lift over one candidate scope."""
    candidates = np.asarray(sorted(set(int(i) for i in candidate_indices)), dtype=int)
    labels = np.isin(candidates, np.asarray(list(positive_indices), dtype=int)).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return dict(n_candidates=int(len(candidates)), n_positives=int(labels.sum()),
                    floor=None, ap=None, lift=None, auroc=None)
    values = np.asarray(scores, dtype=float)[candidates]
    floor = float(labels.mean())
    ap = average_precision(labels, values)
    return dict(n_candidates=int(len(candidates)), n_positives=int(labels.sum()),
                floor=floor, ap=ap, lift=float(ap / floor),
                auroc=auroc(labels, values))


def hits_at_k(scores, positive_indices, candidate_indices, k=5):
    """Return (hits, k, chosen indices) taking the top ``k`` of the scope."""
    candidates = np.asarray(sorted(set(int(i) for i in candidate_indices)), dtype=int)
    values = np.asarray(scores, dtype=float)[candidates]
    chosen = candidates[np.argsort(-values, kind="stable")[:k]]
    positives = set(int(i) for i in positive_indices)
    return int(sum(1 for i in chosen if int(i) in positives)), int(k), [int(i) for i in chosen]


def control_scores(coordinates, adjacency, seed_indices, positive_indices):
    """The three deterministic self-check scores, run through the identical path.

    The random control is handled separately by :func:`mean_random_lift`: a
    single random draw has large variance when the matched scope holds only a
    few dozen residues, so one draw does not estimate the floor.
    """
    xyz = np.asarray(coordinates, dtype=float)
    perfect = np.isin(np.arange(len(xyz)),
                      np.asarray(list(positive_indices), dtype=int)).astype(float)
    return dict(
        perfect=perfect,
        burial=burial(xyz),
        neg_seed_dist=-seed_hops(adjacency, seed_indices),
    )


def mean_random_lift(node_count, positive_indices, candidate_indices, draws=200):
    """Mean lift of a random score over ``draws`` independent draws.

    Reported instead of a single draw because the matched scope is small: with a
    few dozen residues the AP of one random vector scatters widely, and a lone
    draw above 1 would be read as a bias that is not there. The standard
    deviation is returned alongside so the spread is visible.
    """
    lifts = []
    for seed in range(int(draws)):
        vector = np.random.default_rng(seed).random(int(node_count))
        result = scope_metrics(vector, positive_indices, candidate_indices)
        if result["lift"] is not None:
            lifts.append(result["lift"])
    if not lifts:
        return None, None
    return float(np.mean(lifts)), float(np.std(lifts))
