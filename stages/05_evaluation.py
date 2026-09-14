"""Stage 05 - three-tier evaluation, from the weakest control to the strictest.

Reads   : 01_sites/<target>_nodes.csv, 02_network/adjacency.npy,
          04_pockets/residue_scores.csv
Writes  : 05_evaluation/evaluation.json, 05_evaluation/ranking.csv,
          05_evaluation/fake_pockets.csv

Tier 1 - **random background.** Positives against every non-seed residue. The
floor is the positive prevalence. This tier is easy to pass and therefore weak
evidence: an allosteric site is buried and distant from the active site by
definition, so a score that measures nothing else already beats it.

Tier 2 - **condition-matched fake pockets.** Positives against the pooled
residues of control pockets whose centres match the true site in burial and in
hop distance from the seed. Two separate hard gates, never summed. Read the
four control scores first: ``perfect`` should be far above 1, and ``random``,
``burial`` and ``neg_seed_dist`` should sit near 1. If ``burial`` or
``neg_seed_dist`` exceeds 1.5 the gate admitted a skewed control set and the
score of interest is being rewarded for geometry - the result is then not
interpretable and the run says so.

Tier 3 - **hits at 5.** How many of the submitted five residues are ground
truth, in the challenge output format.

Seed residues are excluded from scoring throughout, because counting a seed as
a hit rewards a method for finding its own starting point. ``shared`` residues
contact both binding partners and are therefore seeds; ``--shared-policy``
records which way they were counted.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER
from qwallo.constants import TOP_K
from qwallo.evaluation import (control_scores, fake_pockets, hits_at_k,
                              mean_random_lift, scope_metrics)
from qwallo.workspace import (read_csv, stage_dir, targets_from_argument,
                             write_csv, write_json)

CONTROL_LIFT_LIMIT = 1.5


def load(work, target):
    nodes = read_csv(stage_dir(work, target, 1) / (target + "_nodes.csv"))
    scores = read_csv(stage_dir(work, target, 4) / "residue_scores.csv")
    if len(nodes) != len(scores):
        raise SystemExit("node table and score table disagree on residue count")
    keys = [row["residue_id"] for row in nodes]
    xyz = np.asarray([[float(row["x"]), float(row["y"]), float(row["z"])]
                      for row in nodes])
    value = np.asarray([float(row["quantum_score"]) for row in scores])
    classes = [row["site_class"] for row in nodes]
    return keys, xyz, value, classes


def evaluate(value, xyz, adjacency, seeds, positives, candidates, control, top_k):
    tier1 = scope_metrics(value, positives, candidates)
    fakes, report = fake_pockets(xyz, adjacency, seeds, positives)
    tier2 = (scope_metrics(value, positives, sorted(set(positives) | set(fakes.tolist())))
             if report["evaluable"] else None)
    controls, random_control = {}, (None, None)
    if report["evaluable"]:
        scope = sorted(set(positives) | set(fakes.tolist()))
        for name, vector in control.items():
            controls[name] = scope_metrics(vector, positives, scope)
        random_control = mean_random_lift(len(value), positives, scope)
    hits, k, chosen = hits_at_k(value, positives, candidates, k=top_k)
    return tier1, tier2, controls, report, fakes, (hits, k, chosen), random_control


def run(target, work, shared_policy, top_k):
    keys, xyz, value, classes = load(work, target)
    adjacency = np.load(stage_dir(work, target, 2) / "adjacency.npy")
    seeds = [i for i, c in enumerate(classes) if c in ("orthosteric", "shared")]
    if shared_policy == "positive":
        positives = [i for i, c in enumerate(classes) if c in ("allosteric", "shared")]
    else:
        positives = [i for i, c in enumerate(classes) if c == "allosteric"]
    candidates = [i for i in range(len(keys)) if i not in set(seeds)]
    if not positives:
        raise SystemExit(target + ": no allosteric residue survives the shared policy")

    control = control_scores(xyz, adjacency, seeds, positives)
    (tier1, tier2, controls, report, fakes, (hits, k, chosen),
     random_control) = evaluate(
        value, xyz, adjacency, seeds, positives, candidates, control, top_k)

    warnings = []
    for name in ("burial", "neg_seed_dist"):
        lift = (controls.get(name) or {}).get("lift")
        if lift is not None and lift > CONTROL_LIFT_LIMIT:
            warnings.append("control_lift_" + name + " = " + format(lift, ".2f")
                            + " exceeds " + str(CONTROL_LIFT_LIMIT)
                            + "; tier 2 is not interpretable")
    if not report["evaluable"]:
        warnings.append("fewer than 20 matched control residues; tier 2 not evaluable")
    for name in ("smd_burial", "smd_seed_hop"):
        if report.get(name) is not None and abs(report[name]) > 1.0:
            warnings.append(name + " = " + format(report[name], ".2f")
                            + " exceeds 1.0 sd; read tier 2 with the control lifts")

    out = stage_dir(work, target, 5, create=True)
    order = np.argsort(-value, kind="stable")
    rank = np.empty(len(value), dtype=int)
    rank[order] = np.arange(1, len(value) + 1)
    write_csv(out / "ranking.csv",
              ("residue_id", "site_class", "score", "rank", "percentile"),
              [(keys[i], classes[i], float(value[i]), int(rank[i]),
                float(1.0 - (rank[i] - 1) / len(value))) for i in order])
    write_csv(out / "fake_pockets.csv", ("residue_id", "matrix_index"),
              [(keys[i], int(i)) for i in fakes])
    write_json(out / "evaluation.json", dict(
        target=target, shared_policy=shared_policy, top_k=top_k,
        n_nodes=len(keys), n_seeds=len(seeds), n_positives=len(positives),
        tier1_random_background=tier1,
        tier2_matched_fake_pockets=tier2,
        tier2_generation=report,
        tier2_control_lifts={name: (m or {}).get("lift") for name, m in controls.items()},
        tier2_random_control=dict(mean_lift=random_control[0], sd=random_control[1],
                                  draws=200),
        tier3_hits_at_k=dict(hits=hits, k=k, residues=[keys[i] for i in chosen]),
        warnings=warnings,
    ))
    line = (target + ": tier1 lift " + format(tier1["lift"], ".2f")
            + "  tier2 lift " + (format(tier2["lift"], ".2f") if tier2 else "n/a")
            + "  hits " + str(hits) + "/" + str(k))
    print(line + ("   [" + str(len(warnings)) + " warning(s)]" if warnings else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--shared-policy", default="exclude",
                        choices=("exclude", "positive"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work, args.shared_policy, args.top_k)


if __name__ == "__main__":
    main()
