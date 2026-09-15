"""Predict a distal residue subset with apo ENM cavities, CTQW, and lining QUBO."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from connectivity.maps import time_averaged_ctqw_connectivity
from connectivity.pipeline import (
    DEFAULT_TIMES, POCKET_TIMES, CTQW_DECAY_TIME, resolve_sources,
    residue_id, select_distal_sources, local_sequence_mask, hop_normalized_source_scores,
)
from connectivity.pockets import Pocket, find_stable_ensemble_pockets, rank_stable_pockets
from connectivity.structure import parse_mmcif_residues, build_uniform_contact_network
from connectivity.subset import build_lining_qubo, solve_pocket_qubo, subset_scores


def residue_range(value):
    try:
        chain, interval = value.split(":", 1)
        start, end = map(int, interval.split("-", 1))
        if not chain or start > end:
            raise ValueError
        return chain, (start, end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must be CHAIN:START-END") from exc


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mmcif", type=Path)
    result.add_argument("--chain", action="append", required=True,
                        help="repeat for a multimer domain")
    result.add_argument("--range", type=residue_range, action="append")
    result.add_argument("--orthosteric", action="append", required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--pair-weight", type=float, default=0.1,
                        help="positive penalizes pairs; negative rewards pairs")
    result.add_argument("--top-k", type=int, default=5)
    result.add_argument("--source-span", type=int, default=10)
    result.add_argument("--pocket-modes", type=int, default=32)
    result.add_argument("--replay", type=Path,
                        help="reuse this run's fixed selected pocket and connectivity")
    return result


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


def replay_pocket(run_dir, nodes, source_ids, input_hash):
    """Read prediction artifacts only; evaluation files are never opened here."""
    metadata = json.loads((run_dir / "method.json").read_text())
    with (run_dir / "residue_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [residue_id(n) for n in nodes]
    saved = [f"{r['chain_id']}:{r['residue_number']}{r['insertion_code']}" for r in rows]
    if keys != saved:
        raise ValueError("replay residue order differs from input structure")
    if metadata.get("input_sha256", input_hash) != input_hash:
        raise ValueError("replay structure hash differs from input")
    if set(source_ids) != set(metadata["orthosteric_candidates"]):
        raise ValueError("replay orthosteric annotation differs from input")
    lookup = {key: i for i, key in enumerate(keys)}
    record = metadata["pocket"]
    selected = tuple(lookup[key] for key in record["selected_pocket"])
    families = tuple(pocket_from_record(p) for p in record["detected_ensemble_pockets"])
    matrix = np.loadtxt(run_dir / "connectivity.csv", delimiter=",")
    return matrix, selected, families, record.get("pocket_ranking", "saved pocket")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.top_k < 1 or args.pocket_modes < 1 or args.source_span < 0:
        raise ValueError("positive top-k/mode count and nonnegative source span required")
    if not np.isfinite(args.pair_weight):
        raise ValueError("pair-weight must be finite")
    chains = list(dict.fromkeys(args.chain))
    ranges = dict(args.range) if args.range else None
    if ranges is not None and set(ranges) != set(chains):
        raise ValueError("one --range per --chain is required")
    if args.replay and args.replay.resolve() == args.out_dir.resolve():
        raise ValueError("replay must write a separate output directory")
    nodes = parse_mmcif_residues(args.mmcif, chains=chains, residue_ranges=ranges)
    if any(n.insertion_code for n in nodes):
        raise ValueError("cavity residue numbers require a domain without insertion codes")
    keys = [residue_id(n) for n in nodes]
    sources = resolve_sources(nodes, args.orthosteric)
    input_hash = hashlib.sha256(args.mmcif.read_bytes()).hexdigest()
    if args.replay:
        C, eligible, families, pocket_method = replay_pocket(
            args.replay, nodes, args.orthosteric, input_hash)
    else:
        adjacency = build_uniform_contact_network(nodes, cutoff=7.8)
        C = time_averaged_ctqw_connectivity(adjacency, times=DEFAULT_TIMES)
        launch = select_distal_sources(nodes, C, sources, local_span=args.source_span,
                                       concentration_size=5, source_count=1).selected_indices
        hop, explained = hop_normalized_source_scores(C, adjacency, launch)
        numbers = [n.residue_number for n in nodes]
        windows = (ranges if ranges is not None
                   else {chains[0]: (min(numbers), max(numbers))})
        families = find_stable_ensemble_pockets(args.mmcif, windows,
                                                mode_count=args.pocket_modes)
        transport = time_averaged_ctqw_connectivity(adjacency, times=POCKET_TIMES,
                                                   decay_time=CTQW_DECAY_TIME)
        rows = np.asarray([hop_normalized_source_scores(transport, adjacency, [s])[0]
                           for s in sources])
        pocket = rank_stable_pockets(
            nodes, adjacency, families, source_indices=launch, orthosteric_indices=sources,
            excluded_indices=local_sequence_mask(nodes, sources, args.source_span),
            raw_scores=C[list(launch)].mean(axis=0), hop_scores=hop,
            hop_explained_variance=explained, source_raw_scores=transport[list(sources)],
            source_hop_scores=rows, top_k=args.top_k)
        eligible, pocket_method = pocket.selected_pocket, pocket.ranking_method
    Q, ids, features = build_lining_qubo(nodes, eligible, C, sources, families,
                                        top_k=args.top_k, pair_weight=args.pair_weight)
    selected, energy, marginal = solve_pocket_qubo(Q, top_k=args.top_k)
    scores = subset_scores(len(nodes), ids, selected, marginal)
    selected_ids = tuple(ids[i] for i in selected)
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "connectivity.csv", C, delimiter=",")
    np.savetxt(output / "residue_qubo.csv", Q, delimiter=",")
    with (output / "residue_qubo_variables.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("qubo_index", "matrix_index", "residue_id", "lining_percentile"))
        writer.writerows((j, i, keys[i], features[j, 2]) for j, i in enumerate(ids))
    with (output / "residue_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("matrix_index", "chain_id", "residue_number", "insertion_code",
                         "residue_name", "quantum_score", "eligible"))
        writer.writerows((i, n.chain_id, n.residue_number, n.insertion_code, n.residue_name,
                          scores[i], int(i in ids)) for i, n in enumerate(nodes))
    with (output / "hit_list.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("rank", "matrix_index", "residue_id", "score"))
        ordered = sorted(selected_ids, key=lambda i: (-scores[i], i))
        writer.writerows((rank, i, keys[i], scores[i]) for rank, i in enumerate(ordered, 1))
    metadata = {
        "schema_version": 1, "input": str(args.mmcif), "input_sha256": input_hash,
        "execution": "classical exact CTQW and fixed-cardinality enumeration",
        "scope": "fixed-pocket replay" if args.replay else "full apo prediction",
        "orthosteric_candidates": [keys[i] for i in sources],
        "parameters": {"pair_weight": args.pair_weight,
                       "top_k": args.top_k,
                       "source_span": args.source_span if not args.replay else None,
                       "pocket_modes": args.pocket_modes if not args.replay else None},
        "pocket": {"selected_pocket": [keys[i] for i in ids], "pocket_ranking": pocket_method,
                   "detected_ensemble_pockets": [asdict(p) for p in families]},
        "qubo": {"matrix_convention": "upper triangular; E=x^T Q x",
                 "constraint": f"sum(x)={args.top_k}; not encoded in the matrix",
                 "energy": energy, "top_residues": [keys[i] for i in ordered],
                 "score_semantics": "normalized optimal inclusion utility + selected-subset priority; outside -1"},
        "replay_provenance": ({"source": str(args.replay),
                               "source_method_sha256": hashlib.sha256((args.replay / "method.json").read_bytes()).hexdigest()}
                              if args.replay else None),
    }
    (output / "method.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Selected: {', '.join(keys[i] for i in ordered)}; energy={energy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
