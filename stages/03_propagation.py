"""Stage 03 - CTQW propagation and hop normalisation.

Reads   : 00_structures/nodes.csv, 01_sites/sites.json, 02_network/adjacency.npy
Writes  : 03_propagation/connectivity.csv     <- submission artefact 1 (N x N)
          03_propagation/transport.npy        (cavity-ranking matrix, binary)
          03_propagation/scores.csv           (raw and hop-normalised per residue)
          03_propagation/propagation.json

Two CTQW matrices are produced from the same graph, because the two downstream
consumers need different time windows:

* ``connectivity`` on ``DEFAULT_TIMES`` = geomspace(0.02, 20, 120) - the
  submitted matrix and the pair term of the residue objective.
* ``transport`` on ``POCKET_TIMES`` = linspace(0.02, 200, 401) weighted by
  exp(-t / 14) - the cavity ranking, which needs the longer window.

Both use ``H = L / lambda_max``, so time is dimensionless and not physical.
``--time-convention`` selects how the distribution is read:

* ``average``   time average over the grid (reference behaviour)
* ``converged`` the instantaneous distribution at one large time
* ``limit``     the infinite-time average in closed form (eigenspace projectors)

Hop normalisation ranks residues only against others at the same hop distance
from the seed, which removes the graph-distance decay. The reported
``hop_explained_variance`` is the share of log-connectivity variance explained
by hop-shell medians: a high value means the raw score is dominated by distance
from the active site.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER, TARGETS
from qwallo.constants import (CONCENTRATION_SIZE, CTQW_DECAY_TIME, DEFAULT_TIMES,
                             POCKET_TIMES, SOURCE_COUNT, SOURCE_SPAN)
from qwallo.pipeline import hop_normalized_source_scores, select_distal_sources
from qwallo.propagation import time_averaged_ctqw_connectivity
from qwallo.workspace import (read_csv, read_json, stage_dir,
                             targets_from_argument, write_csv, write_json)


def instantaneous(adjacency, time):
    """Single-time CTQW occupancy |U(t)|^2 with the same normalisation."""
    matrix = np.asarray(adjacency, dtype=float)
    laplacian = np.diag(matrix.sum(axis=1)) - matrix
    values, vectors = np.linalg.eigh(laplacian)
    values = values / max(float(values.max()), 1e-12)
    amplitudes = (vectors * np.exp(-1j * values * float(time))) @ vectors.T
    probabilities = np.abs(amplitudes) ** 2
    np.fill_diagonal(probabilities, 0.0)
    return np.clip(0.5 * (probabilities + probabilities.T), 0.0, 1.0)


def infinite_limit(adjacency):
    """Closed-form t -> infinity time average: sum of squared eigenprojectors."""
    matrix = np.asarray(adjacency, dtype=float)
    laplacian = np.diag(matrix.sum(axis=1)) - matrix
    values, vectors = np.linalg.eigh(laplacian)
    probabilities = np.zeros_like(matrix)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(values[end] - values[start]) < 1e-9:
            end += 1
        projector = vectors[:, start:end] @ vectors[:, start:end].T
        probabilities += projector * projector
        start = end
    np.fill_diagonal(probabilities, 0.0)
    return np.clip(0.5 * (probabilities + probabilities.T), 0.0, 1.0)


def connectivity_for(adjacency, convention, times, decay_time, converged_time):
    if convention == "average":
        return time_averaged_ctqw_connectivity(adjacency, times=times,
                                               decay_time=decay_time)
    if convention == "converged":
        return instantaneous(adjacency, converged_time)
    if convention == "limit":
        return infinite_limit(adjacency)
    raise SystemExit("unknown time convention: " + convention)


def run(target, work, convention, converged_time):
    spec = TARGETS[target]
    rows = read_csv(stage_dir(work, target, 0) / "nodes.csv")
    keys = [row["residue_id"] for row in rows]
    lookup = {key: index for index, key in enumerate(keys)}
    sites = read_json(stage_dir(work, target, 1) / "sites.json")
    sources = tuple(lookup[key] for key in sites["active_site"]["residues"])
    adjacency = np.load(stage_dir(work, target, 2) / "adjacency.npy")

    connectivity = connectivity_for(adjacency, convention, DEFAULT_TIMES, None,
                                    converged_time)
    transport = connectivity_for(adjacency, convention, POCKET_TIMES,
                                 CTQW_DECAY_TIME, converged_time)

    class _Node:
        __slots__ = ("chain_id", "residue_number")

        def __init__(self, chain_id, residue_number):
            self.chain_id, self.residue_number = chain_id, residue_number

    nodes = [_Node(row["chain_id"] if "chain_id" in row else spec["chain"],
                   int(row["residue_number"])) for row in rows]
    launch = select_distal_sources(nodes, connectivity, sources,
                                   local_span=SOURCE_SPAN,
                                   concentration_size=CONCENTRATION_SIZE,
                                   source_count=SOURCE_COUNT)
    hop, explained = hop_normalized_source_scores(connectivity, adjacency,
                                                  launch.selected_indices)
    source_rows = np.asarray([
        hop_normalized_source_scores(transport, adjacency, [s])[0] for s in sources
    ])
    raw = connectivity[list(launch.selected_indices)].mean(axis=0)

    out = stage_dir(work, target, 3, create=True)
    np.savetxt(out / "connectivity.csv", connectivity, delimiter=",")
    np.save(out / "transport.npy", transport)
    np.save(out / "source_hop_rows.npy", source_rows)
    write_csv(out / "scores.csv",
              ("matrix_index", "residue_id", "raw_ctqw", "hop_normalised"),
              [(i, keys[i], float(raw[i]), float(hop[i])) for i in range(len(keys))])
    write_json(out / "propagation.json", dict(
        target=target, time_convention=convention,
        converged_time=float(converged_time) if convention == "converged" else None,
        default_times=dict(start=float(DEFAULT_TIMES[0]), stop=float(DEFAULT_TIMES[-1]),
                           count=int(len(DEFAULT_TIMES)), spacing="geometric"),
        pocket_times=dict(start=float(POCKET_TIMES[0]), stop=float(POCKET_TIMES[-1]),
                          count=int(len(POCKET_TIMES)), spacing="linear",
                          decay_time=CTQW_DECAY_TIME),
        hamiltonian="L / lambda_max (normalised Laplacian, dimensionless time)",
        transport_identical_to_connectivity=bool(convention != "average"),
        orthosteric_candidates=list(sites["active_site"]["residues"]),
        distal_concentration=dict(zip(
            [keys[i] for i in launch.candidate_indices],
            [float(v) for v in launch.concentrations])),
        launch_residues=[keys[i] for i in launch.selected_indices],
        hop_explained_variance=float(explained),
    ))
    print(target + ": launch " + ",".join(keys[i] for i in launch.selected_indices)
          + "  hop-explained variance " + format(explained, ".3f"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--time-convention", default="average",
                        choices=("average", "converged", "limit"))
    parser.add_argument("--converged-time", type=float, default=1000.0)
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work, args.time_convention, args.converged_time)


if __name__ == "__main__":
    main()
