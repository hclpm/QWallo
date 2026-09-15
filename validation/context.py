"""Rebuild the evaluation context of one target from the published artefacts.

Nothing here re-predicts. The contact graph, the seed set and the propagation
vector are rebuilt with the same modules the predictor calls, and the selected
five residues are read from ``results/``, so a number in the report and the
number the predictor produced come from one code path.

The scoring scope follows the site definition: seeds are the launch points and
are never scored, and a ground-truth residue that coincides with a seed leaves
the candidate set with it.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectivity.maps import time_averaged_ctqw_connectivity
from connectivity.pipeline import (
    DEFAULT_TIMES, hop_normalized_source_scores, residue_id, resolve_sources,
    select_distal_sources,
)
from connectivity.structure import build_uniform_contact_network, parse_mmcif_residues

from run import TARGETS
from validate import TARGETS as HOLO_TARGETS, ligand_contacts

CONTACT_CUTOFF = 7.8
SOURCE_SPAN = 10


def windows(target):
    """Return (chain list, {chain: (start, end)}) for a preset."""
    _, residue_range, _ = TARGETS[target]
    spec = (residue_range,) if isinstance(residue_range, str) else tuple(residue_range)
    chains, ranges = [], {}
    for window in spec:
        chain, interval = window.split(":", 1)
        start, end = interval.split("-", 1)
        chains.append(chain)
        ranges[chain] = (int(start), int(end))
    return chains, ranges


def nodes_of(target):
    structure, _, _ = TARGETS[target]
    chains, ranges = windows(target)
    return parse_mmcif_residues(ROOT / "data" / target / structure,
                                chains=chains, residue_ranges=ranges)


def predicted_scores(target):
    """The per-residue objective score the predictor wrote for this target."""
    path = ROOT / "results" / target / "residue_scores.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([float(row["quantum_score"]) for row in rows])


def selected_five(target):
    path = ROOT / "results" / target / "hit_list.csv"
    with path.open(newline="") as handle:
        return [row["residue_id"] for row in csv.DictReader(handle)]


def ground_truth(target, keys):
    """Indices of residues contacting the allosteric binding partner in holo."""
    if target not in HOLO_TARGETS:
        return None
    holo, partner = HOLO_TARGETS[target]
    truth = ligand_contacts(ROOT / "validation_data" / target / holo, partner)
    return [i for i, key in enumerate(keys) if key in truth]


def build(target, cutoff=CONTACT_CUTOFF):
    """Return the full evaluation context at one contact cutoff."""
    chains, _ = windows(target)
    _, _, orthosteric = TARGETS[target]
    nodes = nodes_of(target)
    keys = [residue_id(node) for node in nodes]
    xyz = np.asarray([node.coordinate for node in nodes], dtype=float)
    adjacency = build_uniform_contact_network(nodes, cutoff=cutoff)
    seeds = list(resolve_sources(nodes, [chains[0] + ":" + str(n) for n in orthosteric]))
    transport = time_averaged_ctqw_connectivity(adjacency, times=DEFAULT_TIMES)
    launch = list(select_distal_sources(
        nodes, transport, seeds, local_span=SOURCE_SPAN,
        concentration_size=5, source_count=1).selected_indices)
    hop, explained = hop_normalized_source_scores(transport, adjacency, launch)

    truth = ground_truth(target, keys)
    seed_set = set(seeds)
    candidates = [i for i in range(len(keys)) if i not in seed_set]
    positives = None if truth is None else [i for i in truth if i not in seed_set]
    return dict(
        target=target, keys=keys, xyz=xyz, adjacency=adjacency, cutoff=cutoff,
        seeds=seeds, launch=launch, candidates=candidates, positives=positives,
        truth_all=truth, transport=transport, hop=hop, hop_explained=explained,
    )
