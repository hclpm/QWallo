"""Stage 06 - classical and alternative-operator baselines on the same graph.

Reads   : 01_sites/<target>_nodes.csv, 02_network/adjacency.npy,
          03_propagation/propagation.json, 04_pockets/method.json
Writes  : 06_baselines/baselines.csv, 06_baselines/baselines.json

Every baseline is given the identical adjacency matrix, the identical seed set,
and the identical hop normalisation, then is read out through the identical
three-tier evaluation. Two axes are compared:

* ``rwr`` - random walk with restart, restart probability swept over
  0.05 ... 0.50. This is the classical diffusion control: if a quantum walk
  carries no information a classical walk lacks, the two lift values match.
* ``coined`` - discrete-time coined quantum walk, Cesaro time average. A second
  quantum operator, to separate "quantum walk" from "this particular walk".

Scores are compared at the **residue** level over the whole non-seed candidate
set, not inside the selected cavity, because the cavity was chosen using CTQW
transport and feeding a baseline through that choice would not be a control.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER
from qwallo.baselines import coined_connectivity, rwr_connectivity
from qwallo.constants import TOP_K
from qwallo.evaluation import fake_pockets, hits_at_k, scope_metrics
from qwallo.workspace import (read_csv, read_json, stage_dir,
                             targets_from_argument, write_csv, write_json)

RESTARTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def hop_normalise(matrix, adjacency, sources):
    hops = shortest_path(np.asarray(adjacency, dtype=float), directed=False,
                         unweighted=True)
    rows = []
    for source in sources:
        values = np.zeros(len(adjacency))
        for hop in np.unique(hops[source]).astype(int):
            if hop == 0:
                continue
            shell = np.flatnonzero(hops[source] == hop)
            values[shell] = rankdata(matrix[source, shell], method="average") / len(shell)
        rows.append(values)
    return np.asarray(rows).mean(axis=0)


def run(target, work, steps, top_k):
    nodes = read_csv(stage_dir(work, target, 1) / (target + "_nodes.csv"))
    keys = [row["residue_id"] for row in nodes]
    classes = [row["site_class"] for row in nodes]
    xyz = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in nodes])
    adjacency = np.load(stage_dir(work, target, 2) / "adjacency.npy")
    meta = read_json(stage_dir(work, target, 3) / "propagation.json")
    lookup = {key: i for i, key in enumerate(keys)}
    launch = [lookup[key] for key in meta["launch_residues"]]

    seeds = [i for i, c in enumerate(classes) if c in ("orthosteric", "shared")]
    positives = [i for i, c in enumerate(classes) if c == "allosteric"]
    candidates = [i for i in range(len(keys)) if i not in set(seeds)]
    fakes, report = fake_pockets(xyz, adjacency, seeds, positives)
    matched = sorted(set(positives) | set(fakes.tolist())) if report["evaluable"] else None

    ctqw_hop = np.asarray([float(r["hop_normalised"])
                           for r in read_csv(stage_dir(work, target, 3) / "scores.csv")])
    variants = {"ctqw_hop (reference)": ctqw_hop}
    for restart in RESTARTS:
        variants["rwr_hop restart=" + format(restart, ".2f")] = hop_normalise(
            rwr_connectivity(adjacency, restart=restart), adjacency, launch)
    variants["coined_hop cesaro"] = hop_normalise(
        coined_connectivity(adjacency, steps=steps, seeds=launch), adjacency, launch)

    rows, payload = [], {}
    for name, vector in variants.items():
        tier1 = scope_metrics(vector, positives, candidates)
        tier2 = scope_metrics(vector, positives, matched) if matched else None
        hits, k, chosen = hits_at_k(vector, positives, candidates, k=top_k)
        rows.append((target, name, tier1["n_candidates"], tier1["n_positives"],
                     round(tier1["floor"], 6), round(tier1["ap"], 6),
                     round(tier1["lift"], 4),
                     "" if tier2 is None else round(tier2["lift"], 4),
                     hits, k, ";".join(keys[i] for i in chosen)))
        payload[name] = dict(tier1=tier1, tier2=tier2, hits=hits, k=k,
                             top_residues=[keys[i] for i in chosen])

    out = stage_dir(work, target, 6, create=True)
    write_csv(out / "baselines.csv",
              ("target", "method", "n_candidates", "n_positives", "floor", "ap",
               "tier1_lift", "tier2_lift", "hits", "k", "top_residues"), rows)
    write_json(out / "baselines.json", dict(
        target=target, coined_steps=steps, restarts=list(RESTARTS),
        seeds_used=[keys[i] for i in launch],
        note=("scores are compared over the whole non-seed candidate set; the "
              "selected cavity is not used because it was chosen with CTQW"),
        results=payload, tier2_generation=report))
    best = max((v for k_, v in payload.items() if k_.startswith("rwr")),
               key=lambda v: v["tier1"]["lift"])
    print(target + ": ctqw lift " + format(payload["ctqw_hop (reference)"]["tier1"]["lift"], ".2f")
          + "  best rwr lift " + format(best["tier1"]["lift"], ".2f")
          + "  coined lift " + format(payload["coined_hop cesaro"]["tier1"]["lift"], ".2f"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--coined-steps", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work, args.coined_steps, args.top_k)


if __name__ == "__main__":
    main()
