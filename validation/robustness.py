"""Does the published answer survive the choices the literature does not fix?

Three axes, each a full re-run of the affected steps rather than an
interpolation. The cavity and the conformer-ensemble families are held at their
published values in every row, so what is being tested is the residue ranking
*after* cavity selection, not the selection itself.

* **contact cutoff** 4.5-8.5 A. The cavity set does not move here, because the
  detector works on atomic coordinates; only the graph, the propagation and the
  ranking move. Graph statistics are reported alongside, because at the low end
  the graph approaches a chain and the hop shells thin out.
* **time convention** the weighted average over the standard grid, a long dense
  grid, and the instantaneous distribution at a large time. A unitary walk never
  decoheres, so these are three readings of one operator, not successive
  approximations.
* **pair weight** 0.0-0.5. Zero removes the redundancy penalty entirely, which
  isolates how much of the subset is decided by cavity-wall geometry alone.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from connectivity.maps import time_averaged_ctqw_connectivity
from connectivity.pipeline import CTQW_DECAY_TIME, DEFAULT_TIMES, POCKET_TIMES, residue_id
from connectivity.structure import build_uniform_contact_network
from connectivity.subset import build_lining_qubo, solve_pocket_qubo, subset_scores
from predict import pocket_from_record

from validation.context import ROOT, build, nodes_of
from validation.metrics import fake_pockets, hits_at_k, scope_metrics
from validation.run_validation import OUT, TARGET_ORDER, _targets, _write_csv

CUTOFFS = (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.8, 8.0, 8.5)
PAIR_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.5)
PAIR_WEIGHT = 0.1
TOP_K = 5
CONVENTIONS = (
    ("weighted grid (published)", DEFAULT_TIMES, CTQW_DECAY_TIME),
    ("long dense grid", POCKET_TIMES, None),
    ("instantaneous t=200", np.array([200.0]), None),
)


def published(target):
    """Cavity, ensemble families and node keys as the predictor recorded them."""
    nodes = nodes_of(target)
    keys = [residue_id(node) for node in nodes]
    lookup = {key: i for i, key in enumerate(keys)}
    record = json.loads((ROOT / "results" / target / "method.json").read_text())["pocket"]
    cavity = tuple(lookup[key] for key in record["selected_pocket"])
    families = tuple(pocket_from_record(p) for p in record["detected_ensemble_pockets"])
    return nodes, keys, cavity, families


def rank(nodes, cavity, families, connectivity, sources, pair_weight=PAIR_WEIGHT):
    qubo, ids, _ = build_lining_qubo(nodes, cavity, connectivity, sources, families,
                                     top_k=TOP_K, pair_weight=pair_weight)
    selected, energy, marginal = solve_pocket_qubo(qubo, top_k=TOP_K)
    return subset_scores(len(nodes), ids, selected, marginal), float(energy)


def scorer(context):
    """Return a function giving (tier1 lift, tier2 lift, hits, chosen ids)."""
    positives, candidates = context["positives"], context["candidates"]
    fakes, report = fake_pockets(context["xyz"], context["adjacency"],
                                 context["seeds"], positives)
    matched = sorted(set(positives) | set(fakes.tolist())) if report["evaluable"] else None

    def measure(vector):
        tier1 = scope_metrics(vector, positives, candidates)
        tier2 = scope_metrics(vector, positives, matched) if matched else None
        hits, _, chosen = hits_at_k(vector, positives, candidates, k=TOP_K)
        return (round(tier1["lift"], 4),
                "" if tier2 is None else round(tier2["lift"], 4),
                hits, [context["keys"][i] for i in chosen])
    return measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="one preset name, or 'all'")
    args = parser.parse_args()

    cutoff_rows, convention_rows, pair_rows = [], [], []
    for target in _targets(args.target):
        nodes, keys, cavity, families = published(target)
        reference = build(target)
        sources = reference["seeds"]
        measure = scorer(reference)

        for cutoff in CUTOFFS:
            try:
                adjacency = build_uniform_contact_network(nodes, cutoff=cutoff)
            except ValueError as error:
                cutoff_rows.append((target, cutoff, "", "", "", "", "", "", str(error)))
                continue
            degree = np.asarray(adjacency, dtype=float).sum(axis=1)
            components = int(connected_components(csr_matrix(adjacency), directed=False)[0])
            connectivity = time_averaged_ctqw_connectivity(
                adjacency, times=DEFAULT_TIMES, decay_time=CTQW_DECAY_TIME)
            vector, _ = rank(nodes, cavity, families, connectivity, sources)
            lift1, lift2, hits, chosen = measure(vector)
            cutoff_rows.append((target, cutoff, int(adjacency.sum() // 2),
                                round(float(degree.mean()), 2), components,
                                lift1, lift2, hits, ";".join(chosen)))
            print(target, cutoff, "edges", int(adjacency.sum() // 2),
                  "hits", hits, flush=True)

        adjacency = reference["adjacency"]
        for name, times, decay in CONVENTIONS:
            connectivity = time_averaged_ctqw_connectivity(adjacency, times=times,
                                                           decay_time=decay)
            vector, _ = rank(nodes, cavity, families, connectivity, sources)
            lift1, lift2, hits, chosen = measure(vector)
            convention_rows.append((target, name, len(np.atleast_1d(times)),
                                    lift1, lift2, hits, ";".join(chosen)))

        connectivity = time_averaged_ctqw_connectivity(adjacency, times=DEFAULT_TIMES,
                                                       decay_time=CTQW_DECAY_TIME)
        for weight in PAIR_WEIGHTS:
            vector, energy = rank(nodes, cavity, families, connectivity, sources,
                                  pair_weight=weight)
            lift1, lift2, hits, chosen = measure(vector)
            pair_rows.append((target, weight, round(energy, 6), lift1, lift2, hits,
                              ";".join(chosen)))
        print(target, "conventions and pair weights done", flush=True)

    _write_csv(OUT / "cutoff_sweep.csv",
               ("target", "cutoff", "n_edges", "mean_degree", "components",
                "tier1_lift", "tier2_lift", "hits_at_5", "top_five"), cutoff_rows)
    _write_csv(OUT / "time_convention.csv",
               ("target", "convention", "n_times", "tier1_lift", "tier2_lift",
                "hits_at_5", "top_five"), convention_rows)
    _write_csv(OUT / "pair_weight.csv",
               ("target", "pair_weight", "qubo_energy", "tier1_lift", "tier2_lift",
                "hits_at_5", "top_five"), pair_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
