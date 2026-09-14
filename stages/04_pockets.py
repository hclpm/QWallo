"""Stage 04 - cavity ensemble, cavity selection, and the residue subset.

Reads   : 00_structures/nodes.csv, 01_sites/sites.json,
          02_network/adjacency.npy, 03_propagation/{connectivity.csv,
          transport.npy, source_hop_rows.npy, propagation.json}
Writes  : 04_pockets/pocket_families.json, 04_pockets/residue_scores.csv,
          04_pockets/hit_list.csv          <- submission artefact 2 (top 5)
          04_pockets/residue_qubo.csv, 04_pockets/residue_qubo_variables.csv,
          04_pockets/method.json

Three steps, in order:

1. **Cavity ensemble.** 32 low ANM modes are displaced in both directions at
   two RMS amplitudes, giving 129 snapshots including the reference. fpocket
   runs on each. A cavity observed at the first amplitude must reappear at the
   second (Jaccard >= 0.5) to be called stable; stable observations are then
   grouped into families by lining overlap. ``persistence`` is the fraction of
   mode events in which a family was seen. This is what lets a cryptic cavity
   that is closed in the deposited structure be proposed at all.

2. **Cavity selection.** Families are scored by within-hop CTQW transport
   averaged over the whole orthosteric region, not over the single launch
   residue. Persistence enters only as a tie-break whose maximum contribution
   is one candidate-rank interval. fpocket's own score and the cavity volume do
   not participate.

3. **Residue subset.** Inside the selected cavity the objective rewards
   cavity-wall support (the alpha-sphere ``void_support`` percentile) and
   penalises redundant pairs (CTQW pair rank times a Gaussian distance taper,
   weight 0.1). All C(m, 5) subsets are enumerated exactly. Note where the
   signal comes from: the unary term is geometry, and CTQW enters selection and
   the pair term - it is not the primary driver of which five residues win.

``--replay DIR`` skips step 1 and 2 and reuses the cavity recorded in
``DIR/method.json``, which is how a run reproduces a published subset without
fpocket installed.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER, TARGETS
from qwallo.constants import (ANM_CUTOFF, OVERLAP_THRESHOLD, PAIR_WEIGHT,
                             POCKET_AMPLITUDES, POCKET_MODES, SOURCE_SPAN, TOP_K)
from qwallo.pipeline import local_sequence_mask, residue_id, resolve_sources
from qwallo.pockets import Pocket, find_stable_ensemble_pockets, rank_stable_pockets
from qwallo.structure import parse_mmcif_residues
from qwallo.subset import build_lining_qubo, solve_pocket_qubo, subset_scores
from qwallo.workspace import (read_csv, read_json, stage_dir,
                             targets_from_argument, write_csv, write_json)


def pocket_from_record(record):
    def pairs(name):
        value = record.get(name, ())
        return tuple((int(k), float(v)) for k, v in
                     (value.items() if isinstance(value, dict) else value))
    return Pocket(
        number=record["number"],
        geometry_score=record.get("geometry_score", record.get("score", 0)),
        residue_numbers=tuple(record.get("residue_numbers", record.get("residues", ()))),
        volume=record.get("volume", record.get("maximum_volume", 0)),
        persistence=record.get("persistence", 1),
        center=record.get("center"),
        residue_persistence=pairs("residue_persistence"),
        void_support=pairs("void_support"),
        observations=tuple(tuple(v) for v in record.get("observations", ())),
    )


def run(target, work, replay, pair_weight, top_k, modes, fpocket):
    spec = TARGETS[target]
    apo = ROOT / "data" / target / spec["apo"]
    nodes = parse_mmcif_residues(apo, chains=[spec["chain"]],
                                 residue_ranges={spec["chain"]: spec["residue_range"]})
    keys = [residue_id(node) for node in nodes]
    lookup = {key: index for index, key in enumerate(keys)}
    sites = read_json(stage_dir(work, target, 1) / "sites.json")
    sources = tuple(lookup[key] for key in sites["active_site"]["residues"])
    adjacency = np.load(stage_dir(work, target, 2) / "adjacency.npy")
    propagation = stage_dir(work, target, 3)
    connectivity = np.loadtxt(propagation / "connectivity.csv", delimiter=",")
    meta = read_json(propagation / "propagation.json")
    launch = tuple(lookup[key] for key in meta["launch_residues"])

    if replay:
        record = read_json(Path(replay) / "method.json")["pocket"]
        families = tuple(pocket_from_record(p) for p in record["detected_ensemble_pockets"])
        eligible = tuple(lookup[key] for key in record["selected_pocket"])
        ranking = record.get("pocket_ranking", "saved cavity") + " (replayed)"
    else:
        transport = np.load(propagation / "transport.npy")
        source_rows = np.load(propagation / "source_hop_rows.npy")
        hop = np.asarray([float(row["hop_normalised"])
                          for row in read_csv(propagation / "scores.csv")])
        numbers = [node.residue_number for node in nodes]
        families = find_stable_ensemble_pockets(
            apo, spec["chain"], min(numbers), max(numbers), mode_count=modes,
            amplitudes=POCKET_AMPLITUDES, anm_cutoff=ANM_CUTOFF,
            overlap_threshold=OVERLAP_THRESHOLD, executable=fpocket)
        selection = rank_stable_pockets(
            nodes, adjacency, families, source_indices=launch,
            orthosteric_indices=sources,
            excluded_indices=local_sequence_mask(nodes, sources, SOURCE_SPAN),
            raw_scores=connectivity[list(launch)].mean(axis=0), hop_scores=hop,
            hop_explained_variance=float(meta["hop_explained_variance"]),
            source_raw_scores=transport[list(sources)], source_hop_scores=source_rows,
            top_k=top_k)
        eligible, ranking = selection.selected_pocket, selection.ranking_method

    qubo, ids, features = build_lining_qubo(nodes, eligible, connectivity, sources,
                                            families, top_k=top_k,
                                            pair_weight=pair_weight)
    selected, energy, marginal = solve_pocket_qubo(qubo, top_k=top_k)
    scores = subset_scores(len(nodes), ids, selected, marginal)
    ordered = sorted((ids[i] for i in selected), key=lambda i: (-scores[i], i))

    out = stage_dir(work, target, 4, create=True)
    np.savetxt(out / "residue_qubo.csv", qubo, delimiter=",")
    write_csv(out / "residue_qubo_variables.csv",
              ("qubo_index", "matrix_index", "residue_id", "lining_percentile"),
              [(j, i, keys[i], float(features[j, 2])) for j, i in enumerate(ids)])
    write_csv(out / "residue_scores.csv",
              ("matrix_index", "chain_id", "residue_number", "insertion_code",
               "residue_name", "quantum_score", "eligible"),
              [(i, n.chain_id, n.residue_number, n.insertion_code, n.residue_name,
                float(scores[i]), int(i in set(ids))) for i, n in enumerate(nodes)])
    write_csv(out / "hit_list.csv", ("rank", "matrix_index", "residue_id", "score"),
              [(rank, i, keys[i], float(scores[i]))
               for rank, i in enumerate(ordered, 1)])
    write_json(out / "pocket_families.json", dict(
        target=target, family_count=len(families),
        families=[asdict(p) for p in families]))
    write_json(out / "method.json", dict(
        schema_version=2, target=target, input=str(apo),
        input_sha256=hashlib.sha256(apo.read_bytes()).hexdigest(),
        execution="classical exact CTQW and fixed-cardinality enumeration",
        scope="fixed-cavity replay" if replay else "full apo prediction",
        orthosteric_candidates=list(sites["active_site"]["residues"]),
        parameters=dict(pair_weight=pair_weight, top_k=top_k,
                        source_span=SOURCE_SPAN,
                        pocket_modes=None if replay else modes,
                        amplitudes=None if replay else list(POCKET_AMPLITUDES),
                        anm_cutoff=None if replay else ANM_CUTOFF),
        pocket=dict(selected_pocket=[keys[i] for i in ids], pocket_ranking=ranking,
                    family_count=len(families)),
        qubo=dict(matrix_convention="upper triangular; E = x^T Q x",
                  constraint="sum(x) = top_k; not encoded in the matrix",
                  energy=float(energy), top_residues=[keys[i] for i in ordered],
                  score_semantics=("normalised optimal inclusion utility plus "
                                   "selected-subset priority; outside the cavity -1")),
        replay_source=str(replay) if replay else None,
    ))
    print(target + ": cavity " + str(len(ids)) + " residues  ranking " + ranking
          + "  hits " + ",".join(keys[i] for i in ordered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--replay-root", default=None,
                        help="reuse the cavity in <root>/<target>/method.json")
    parser.add_argument("--pair-weight", type=float, default=PAIR_WEIGHT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--pocket-modes", type=int, default=POCKET_MODES)
    parser.add_argument("--fpocket", default="fpocket", help="fpocket executable")
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        replay = (Path(args.replay_root) / target) if args.replay_root else None
        run(target, args.work, replay, args.pair_weight, args.top_k,
            args.pocket_modes, args.fpocket)


if __name__ == "__main__":
    main()
