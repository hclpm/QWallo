"""Score a published prediction and run the two method checks.

Three subcommands, each writing into ``validation_out/``:

``score``      two-tier scoring of the published five residues per target
``baselines``  the same graph and seeds under other propagators
``cutoff``     the same pipeline over a range of contact cutoffs

Tier 1 scores against every residue that is not a seed. Tier 2 replaces that
background with control pockets matched to the true site in burial and in graph
distance from the seed, which is the comparison a reviewer needs: a site can
score well on tier 1 simply for being buried and far from the seed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata

from validation.baselines import coined_connectivity, rwr_connectivity
from validation.context import ROOT, build, predicted_scores, selected_five
from validation.metrics import (control_scores, fake_pockets, hits_at_k,
                                mean_random_lift, scope_metrics)

TARGET_ORDER = ("kras", "bcr", "myosin", "mapk14", "ptpn1", "pdk2", "ptpn11")
RESTARTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CUTOFFS = (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.8, 8.0, 8.5)
CONTROL_LIFT_LIMIT = 1.5
OUT = ROOT / "validation_out"


def _targets(argument):
    return list(TARGET_ORDER) if argument == "all" else [argument]


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def hop_normalise(matrix, adjacency, sources):
    """Rank each residue inside its own hop shell, then average over sources."""
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


def score_target(target):
    context = build(target)
    keys, positives = context["keys"], context["positives"]
    if not positives:
        raise SystemExit(target + ": no scorable ground-truth residue")
    value = predicted_scores(target)
    if len(value) != len(keys):
        raise SystemExit(target + ": score table and node table disagree")

    tier1 = scope_metrics(value, positives, context["candidates"])
    fakes, report = fake_pockets(context["xyz"], context["adjacency"],
                                 context["seeds"], positives)
    matched = sorted(set(positives) | set(fakes.tolist())) if report["evaluable"] else None
    tier2 = scope_metrics(value, positives, matched) if matched else None

    controls, random_control = {}, (None, None)
    if matched:
        for name, vector in control_scores(context["xyz"], context["adjacency"],
                                           context["seeds"], positives).items():
            controls[name] = scope_metrics(vector, positives, matched)
        random_control = mean_random_lift(len(value), positives, matched)

    hits, k, chosen = hits_at_k(value, positives, context["candidates"], k=5)
    published = selected_five(target)
    if [keys[i] for i in chosen] != published:
        raise SystemExit(target + ": recomputed top five disagrees with results/")

    warnings = []
    for name in ("burial", "neg_seed_dist"):
        lift = (controls.get(name) or {}).get("lift")
        if lift is not None and lift > CONTROL_LIFT_LIMIT:
            warnings.append("control lift " + name + " = " + format(lift, ".2f")
                            + " exceeds " + str(CONTROL_LIFT_LIMIT)
                            + "; tier 2 is not interpretable for this target")
    if not report["evaluable"]:
        warnings.append("fewer than 20 matched control residues; tier 2 not evaluable")

    payload = dict(target=target, contact_cutoff=context["cutoff"],
                   n_residues=len(keys), n_seeds=len(context["seeds"]),
                   n_candidates=len(context["candidates"]),
                   n_truth_total=len(context["truth_all"]),
                   n_truth_scored=len(positives),
                   launch=[keys[i] for i in context["launch"]],
                   tier1=tier1, tier2=tier2, matched_controls=report,
                   control_lifts={n: c["lift"] for n, c in controls.items()},
                   random_control_lift=random_control[0],
                   hits_at_5=hits, top_five=published, warnings=warnings)
    _write_json(OUT / target / "evaluation.json", payload)
    _write_csv(OUT / target / "fake_pockets.csv", ("residue_id", "matrix_index"),
               [(keys[i], int(i)) for i in fakes])
    return payload


def cmd_score(args):
    rows = []
    for target in _targets(args.target):
        p = score_target(target)
        t2 = p["tier2"] or {}
        rows.append((target, p["n_residues"], p["n_seeds"], p["n_candidates"],
                     p["n_truth_total"], p["n_truth_scored"],
                     round(p["tier1"]["floor"], 6), round(p["tier1"]["ap"], 6),
                     round(p["tier1"]["lift"], 3), round(p["tier1"]["auroc"], 4),
                     t2.get("n_candidates", ""), "" if not t2 else round(t2["lift"], 3),
                     p["hits_at_5"], ";".join(p["top_five"]),
                     " | ".join(p["warnings"])))
        print(target, "tier1 lift", round(p["tier1"]["lift"], 2),
              "| tier2 lift", "-" if not t2 else round(t2["lift"], 2),
              "| hits", p["hits_at_5"], flush=True)
    _write_csv(OUT / "scoring.csv",
               ("target", "n_residues", "n_seeds", "n_candidates", "n_truth_total",
                "n_truth_scored", "tier1_floor", "tier1_ap", "tier1_lift", "auroc",
                "tier2_n_candidates", "tier2_lift", "hits_at_5", "top_five", "warnings"),
               rows)


def cmd_baselines(args):
    rows = []
    for target in _targets(args.target):
        context = build(target)
        keys, positives = context["keys"], context["positives"]
        adjacency, launch = context["adjacency"], context["launch"]
        fakes, report = fake_pockets(context["xyz"], adjacency, context["seeds"], positives)
        matched = sorted(set(positives) | set(fakes.tolist())) if report["evaluable"] else None

        variants = {"ctqw_hop (reference)": context["hop"]}
        for restart in RESTARTS:
            variants["rwr_hop restart=" + format(restart, ".2f")] = hop_normalise(
                rwr_connectivity(adjacency, restart=restart), adjacency, launch)
        variants["coined_hop cesaro"] = hop_normalise(
            coined_connectivity(adjacency, steps=args.steps, seeds=launch),
            adjacency, launch)

        for name, vector in variants.items():
            tier1 = scope_metrics(vector, positives, context["candidates"])
            tier2 = scope_metrics(vector, positives, matched) if matched else None
            hits, _, chosen = hits_at_k(vector, positives, context["candidates"], k=5)
            rows.append((target, name, tier1["n_candidates"], tier1["n_positives"],
                         round(tier1["floor"], 6), round(tier1["ap"], 6),
                         round(tier1["lift"], 4),
                         "" if tier2 is None else round(tier2["lift"], 4),
                         hits, ";".join(keys[i] for i in chosen)))
        print(target, "done", flush=True)
    _write_csv(OUT / "baselines.csv",
               ("target", "method", "n_candidates", "n_positives", "floor", "ap",
                "tier1_lift", "tier2_lift", "hits_at_5", "top_five"), rows)


def cmd_cutoff(args):
    rows = []
    for target in _targets(args.target):
        lining = set(json.loads((ROOT / "results" / target / "method.json").read_text()
                                )["pocket"]["selected_pocket"])
        for cutoff in CUTOFFS:
            context = build(target, cutoff=cutoff)
            keys, positives = context["keys"], context["positives"]
            adjacency = context["adjacency"]
            degree = np.asarray(adjacency, dtype=float).sum(axis=1)
            inside = [i for i, key in enumerate(keys) if key in lining
                      and i not in set(context["seeds"])]
            hits, _, chosen = hits_at_k(context["hop"], positives, inside, k=5)
            rows.append((target, cutoff, len(keys), int(adjacency.sum() // 2),
                         round(float(degree.mean()), 3), int((degree == 0).sum()),
                         hits, ";".join(keys[i] for i in chosen)))
            print(target, cutoff, "edges", int(adjacency.sum() // 2), "hits", hits, flush=True)
    _write_csv(OUT / "cutoff_sweep.csv",
               ("target", "cutoff", "n_residues", "n_edges", "mean_degree",
                "isolated_nodes", "hits_at_5", "top_five"), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("score", cmd_score), ("baselines", cmd_baselines),
                          ("cutoff", cmd_cutoff)):
        p = sub.add_parser(name)
        p.add_argument("target", help="one preset name, or 'all'")
        if name == "baselines":
            p.add_argument("--steps", type=int, default=1000)
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
