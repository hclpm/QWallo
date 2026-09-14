"""Stage 02 - build the residue contact network and its Laplacian.

Reads   : 00_structures/nodes.csv
Writes  : 02_network/adjacency.npy, 02_network/network.json

One node per residue placed at its C-alpha. An edge exists when two C-alpha
atoms lie within 7.8 A. Edges are undirected, unweighted, and the diagonal is
zero. No sequence-adjacency filter is applied: consecutive residues fall inside
the cutoff automatically, and removing them would fragment the graph.

An isolated node is rejected rather than tolerated, because hop normalisation
and the CTQW spectrum both assume a connected graph.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER
from qwallo.constants import CONTACT_CUTOFF
from qwallo.structure import build_uniform_contact_network, parse_mmcif_residues
from qwallo.workspace import stage_dir, targets_from_argument, write_json
from config.targets import TARGETS


def load_nodes(target):
    spec = TARGETS[target]
    return parse_mmcif_residues(
        ROOT / "data" / target / spec["apo"], chains=[spec["chain"]],
        residue_ranges={spec["chain"]: spec["residue_range"]},
    )


def run(target, work, cutoff):
    nodes = load_nodes(target)
    adjacency = build_uniform_contact_network(nodes, cutoff=cutoff)
    degree = adjacency.sum(axis=1)
    components = int(connected_components(csr_matrix(adjacency), directed=False)[0])
    out = stage_dir(work, target, 2, create=True)
    np.save(out / "adjacency.npy", adjacency)
    write_json(out / "network.json", dict(
        target=target, cutoff_angstrom=float(cutoff), node_definition="C-alpha",
        edge_definition="C-alpha pair distance <= cutoff, binary, undirected",
        sequence_adjacency_filter=False,
        nodes=int(len(adjacency)), edges=int(adjacency.sum() // 2),
        mean_degree=float(degree.mean()), min_degree=int(degree.min()),
        components=components, isolated_nodes=int((degree == 0).sum()),
    ))
    print(target + ": " + str(len(adjacency)) + " nodes  "
          + str(int(adjacency.sum() // 2)) + " edges  mean degree "
          + format(degree.mean(), ".2f") + "  components " + str(components))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF)
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work, args.cutoff)


if __name__ == "__main__":
    main()
