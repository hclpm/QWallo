"""Stage 07 - does the result survive the choices that were not forced?

Reads   : 00_structures/nodes.csv, 01_sites/<target>_nodes.csv,
          04_pockets/method.json (for the selected cavity)
Writes  : 07_robustness/cutoff_sweep.csv, 07_robustness/time_convention.csv,
          07_robustness/pair_weight.csv, 07_robustness/robustness.json

Three axes, each one a parameter that literature does not fix:

* **contact cutoff** 4.5 - 8.5 A. The cavity set is invariant here, because
  fpocket works on atomic coordinates, so only the propagation and the ranking
  move. Graph statistics are reported alongside: at the low end the graph
  approaches a chain, the hop shells thin out, and hop normalisation loses the
  population it ranks within.
* **time convention** - time average over the grid, the instantaneous
  distribution at a large time, and the closed-form infinite-time average. A
  unitary walk never decoheres, so these are three different readings of the
  same operator rather than successive approximations.
* **pair weight** 0.0 - 0.5. Zero removes the redundancy penalty entirely,
  which isolates how much of the subset is decided by cavity-wall geometry
  alone.

Every row is a full re-run of the affected stages, not an interpolation.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stages"))

from config.targets import ORDER, TARGETS
from qwallo.constants import (CONCENTRATION_SIZE, CONTACT_CUTOFF, CTQW_DECAY_TIME,
                             DEFAULT_TIMES, PAIR_WEIGHT, POCKET_TIMES,
                             SOURCE_COUNT, SOURCE_SPAN, TOP_K)
from qwallo.evaluation import fake_pockets, hits_at_k, scope_metrics
from qwallo.pipeline import (hop_normalized_source_scores, residue_id,
                            select_distal_sources)
from qwallo.propagation import time_averaged_ctqw_connectivity
from qwallo.structure import build_uniform_contact_network, parse_mmcif_residues
from qwallo.subset import build_lining_qubo, solve_pocket_qubo, subset_scores
from qwallo.workspace import (read_csv, read_json, stage_dir,
                             targets_from_argument, write_csv, write_json)

CUTOFFS = (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5)
PAIR_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.5)


def _context(work, target):
    spec = TARGETS[target]
    nodes = parse_mmcif_residues(ROOT / "data" / target / spec["apo"],
                                 chains=[spec["chain"]],
                                 residue_ranges={spec["chain"]: spec["residue_range"]})
    keys = [residue_id(node) for node in nodes]
    lookup = {key: i for i, key in enumerate(keys)}
    table = read_csv(stage_dir(work, target, 1) / (target + "_nodes.csv"))
    classes = [row["site_class"] for row in table]
    xyz = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in table])
    method = read_json(stage_dir(work, target, 4) / "method.json")
    cavity = tuple(lookup[key] for key in method["pocket"]["selected_pocket"])
    stage4 = importlib.import_module("04_pockets")
    families = tuple(stage4.pocket_from_record(record) for record in
                     read_json(stage_dir(work, target, 4)
                               / "pocket_families.json")["families"])
    sites = read_json(stage_dir(work, target, 1) / "sites.json")
    sources = tuple(lookup[key] for key in sites["active_site"]["residues"])
    seeds = [i for i, c in enumerate(classes) if c in ("orthosteric", "shared")]
    positives = [i for i, c in enumerate(classes) if c == "allosteric"]
    return nodes, keys, xyz, classes, cavity, families, sources, seeds, positives


def _score(nodes, adjacency, connectivity, cavity, families, sources, pair_weight, top_k):
    qubo, ids, _ = build_lining_qubo(nodes, cavity, connectivity, sources, families,
                                     top_k=top_k, pair_weight=pair_weight)
    selected, energy, marginal = solve_pocket_qubo(qubo, top_k=top_k)
    return subset_scores(len(nodes), ids, selected, marginal), float(energy)


def run(target, work, top_k):
    (nodes, keys, xyz, classes, cavity, families, sources,
     seeds, positives) = _context(work, target)
    candidates = [i for i in range(len(keys)) if i not in set(seeds)]
    fakes, report = fake_pockets(xyz, np.load(stage_dir(work, target, 2)
                                              / "adjacency.npy"), seeds, positives)
    matched = sorted(set(positives) | set(fakes.tolist())) if report["evaluable"] else None

    def metrics(vector):
        tier1 = scope_metrics(vector, positives, candidates)
        tier2 = scope_metrics(vector, positives, matched) if matched else None
        hits, k, chosen = hits_at_k(vector, positives, candidates, k=top_k)
        return tier1, tier2, hits, k, chosen

    cutoff_rows = []
    for cutoff in CUTOFFS:
        try:
            adjacency = build_uniform_contact_network(nodes, cutoff=cutoff)
        except ValueError as error:
            cutoff_rows.append((target, cutoff, "", "", "", "", "", "", "", str(error)))
            continue
        degree = adjacency.sum(axis=1)
        components = int(connected_components(csr_matrix(adjacency), directed=False)[0])
        connectivity = time_averaged_ctqw_connectivity(adjacency, times=DEFAULT_TIMES)
        vector, _ = _score(nodes, adjacency, connectivity, cavity, families,
                           sources, PAIR_WEIGHT, top_k)
        tier1, tier2, hits, k, _ = metrics(vector)
        cutoff_rows.append((target, cutoff, int(adjacency.sum() // 2),
                            round(float(degree.mean()), 2), components,
                            round(tier1["lift"], 4),
                            "" if tier2 is None else round(tier2["lift"], 4),
                            hits, k, ""))

    adjacency = build_uniform_contact_network(nodes, cutoff=CONTACT_CUTOFF)
    convention_rows = []
    stage3 = importlib.import_module("03_propagation")
    for convention in ("average", "converged", "limit"):
        connectivity = stage3.connectivity_for(adjacency, convention, DEFAULT_TIMES,
                                               None, 1000.0)
        vector, _ = _score(nodes, adjacency, connectivity, cavity, families,
                           sources, PAIR_WEIGHT, top_k)
        tier1, tier2, hits, k, chosen = metrics(vector)
        convention_rows.append((target, convention, round(tier1["lift"], 4),
                                "" if tier2 is None else round(tier2["lift"], 4),
                                hits, k, ";".join(keys[i] for i in chosen)))

    connectivity = time_averaged_ctqw_connectivity(adjacency, times=DEFAULT_TIMES)
    weight_rows = []
    for weight in PAIR_WEIGHTS:
        vector, energy = _score(nodes, adjacency, connectivity, cavity, families,
                                sources, weight, top_k)
        tier1, tier2, hits, k, chosen = metrics(vector)
        weight_rows.append((target, weight, round(energy, 6), round(tier1["lift"], 4),
                            "" if tier2 is None else round(tier2["lift"], 4),
                            hits, k, ";".join(keys[i] for i in chosen)))

    out = stage_dir(work, target, 7, create=True)
    write_csv(out / "cutoff_sweep.csv",
              ("target", "cutoff", "edges", "mean_degree", "components",
               "tier1_lift", "tier2_lift", "hits", "k", "error"), cutoff_rows)
    write_csv(out / "time_convention.csv",
              ("target", "convention", "tier1_lift", "tier2_lift", "hits", "k",
               "top_residues"), convention_rows)
    write_csv(out / "pair_weight.csv",
              ("target", "pair_weight", "energy", "tier1_lift", "tier2_lift",
               "hits", "k", "top_residues"), weight_rows)
    write_json(out / "robustness.json", dict(
        target=target, cutoffs=list(CUTOFFS), pair_weights=list(PAIR_WEIGHTS),
        conventions=["average", "converged", "limit"],
        cavity_is_cutoff_invariant=True,
        note=("the cavity set comes from fpocket on atomic coordinates, so a "
              "cutoff change moves only the propagation and the ranking"),
        tier2_generation=report))
    print(target + ": cutoff hits " + ",".join(str(r[7]) for r in cutoff_rows)
          + "  convention hits " + ",".join(str(r[4]) for r in convention_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work, args.top_k)


if __name__ == "__main__":
    main()
