"""Stage 08 - collect every stage into one table and one markdown report.

Reads   : all stage outputs under the work directory
Writes  : 08_report/summary.csv, 08_report/reproduction.csv,
          08_report/report.md

``reproduction.csv`` is written only when ``--reference`` points at a directory
of published ``<target>/`` results. It compares the submitted five residues and
the hits count element by element, so a difference is visible per target rather
than as one aggregate pass or fail.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER, TARGETS
from qwallo.workspace import read_csv, read_json, write_csv


def collect(work, target):
    network = read_json(Path(work) / target / "02_network" / "network.json")
    propagation = read_json(Path(work) / target / "03_propagation" / "propagation.json")
    method = read_json(Path(work) / target / "04_pockets" / "method.json")
    sites = read_json(Path(work) / target / "01_sites" / "sites.json")
    evaluation = read_json(Path(work) / target / "05_evaluation" / "evaluation.json")
    hits = read_csv(Path(work) / target / "04_pockets" / "hit_list.csv")
    tier1 = evaluation["tier1_random_background"]
    tier2 = evaluation["tier2_matched_fake_pockets"] or {}
    return dict(
        target=target, name=TARGETS[target]["name"],
        nodes=network["nodes"], edges=network["edges"],
        mean_degree=network["mean_degree"], components=network["components"],
        seeds=evaluation["n_seeds"], positives=evaluation["n_positives"],
        launch=";".join(propagation["launch_residues"]),
        hop_explained=round(propagation["hop_explained_variance"], 4),
        cavity_size=len(method["pocket"]["selected_pocket"]),
        cavity_ranking=method["pocket"]["pocket_ranking"],
        scope=method["scope"],
        tier1_floor=round(tier1["floor"], 6), tier1_ap=round(tier1["ap"], 6),
        tier1_lift=round(tier1["lift"], 4),
        tier2_negatives=evaluation["tier2_generation"]["n_negatives"],
        tier2_lift=round(tier2["lift"], 4) if tier2.get("lift") is not None else "",
        control_burial_lift=(round(evaluation["tier2_control_lifts"]["burial"], 4)
                             if evaluation["tier2_control_lifts"].get("burial") else ""),
        control_seeddist_lift=(round(evaluation["tier2_control_lifts"]["neg_seed_dist"], 4)
                               if evaluation["tier2_control_lifts"].get("neg_seed_dist") else ""),
        hits=evaluation["tier3_hits_at_k"]["hits"],
        k=evaluation["tier3_hits_at_k"]["k"],
        top_residues=";".join(row["residue_id"] for row in hits),
        warnings=" | ".join(evaluation["warnings"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default=str(ROOT / "work"))
    parser.add_argument("--reference", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(args.work) / "08_report"
    out.mkdir(parents=True, exist_ok=True)

    collected = []
    for target in ORDER:
        try:
            collected.append(collect(args.work, target))
        except FileNotFoundError as error:
            print("skip " + target + ": " + str(error))
    if not collected:
        raise SystemExit("no completed target found under " + args.work)
    header = tuple(collected[0])
    write_csv(out / "summary.csv", header,
              [tuple(row[key] for key in header) for row in collected])

    if args.reference:
        rows = []
        for row in collected:
            reference = Path(args.reference) / row["target"]
            hit_file = reference / "hit_list.csv"
            evaluation_file = reference / "evaluation.json"
            published = (";".join(r["residue_id"] for r in read_csv(hit_file))
                         if hit_file.exists() else "")
            published_hits = (read_json(evaluation_file)["metrics"]["hits_at_5"]
                              if evaluation_file.exists() else "")
            rows.append((row["target"], published, row["top_residues"],
                         int(published == row["top_residues"]),
                         published_hits, row["hits"],
                         int(published_hits == row["hits"])
                         if published_hits != "" else ""))
        write_csv(out / "reproduction.csv",
                  ("target", "published_top5", "rebuilt_top5", "top5_identical",
                   "published_hits", "rebuilt_hits", "hits_identical"), rows)

    lines = ["# Run report", "",
             "| target | nodes | edges | seeds | positives | cavity | tier1 lift | "
             "tier2 lift | hits |", "|---|---|---|---|---|---|---|---|---|"]
    for row in collected:
        lines.append("| " + " | ".join(str(row[key]) for key in (
            "target", "nodes", "edges", "seeds", "positives", "cavity_size",
            "tier1_lift", "tier2_lift", "hits")) + " |")
    total = sum(row["hits"] for row in collected)
    lines += ["", "Hits at 5 summed over " + str(len(collected)) + " targets: "
              + str(total) + "/" + str(5 * len(collected)), ""]
    flagged = [row for row in collected if row["warnings"]]
    if flagged:
        lines.append("## Warnings")
        for row in flagged:
            lines.append("- **" + row["target"] + "** " + row["warnings"])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote " + str(out / "summary.csv") + " and " + str(out / "report.md"))


if __name__ == "__main__":
    main()
