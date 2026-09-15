"""Assemble challenge outputs 1 and 2 from the published results.

The challenge statement fixes the content of both outputs and leaves the file
format open, so the convention is declared here and written into the bundle:

  Connectivity matrix   one CSV per target, labelled on both axes with the
                        residue id, so entry (i, j) is readable without a
                        second file. Symmetric, zero diagonal, ten significant
                        digits.
  Hit list              one CSV per target plus a combined file, ranked 1-5,
                        carrying the residue id in the same notation the matrix
                        axes use.

Nothing is recomputed. Values are read from results/<target>/ as published.
"""

import csv
import json
from pathlib import Path

import numpy as np

from validation.context import build

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission"
TARGETS = [("kras", "KRAS G12C", "4OBE"), ("bcr", "BCR-ABL1", "1OPL"),
           ("myosin", "Cardiac myosin", "5TBY"), ("cmyc", "c-Myc/Max", "1NKP")]
PRECISION = "%.10g"


def residue_table(target):
    rows = list(csv.DictReader(open(ROOT / "results" / target / "residue_scores.csv")))
    ids = [r["chain_id"] + ":" + r["residue_number"] + r["insertion_code"].strip()
           for r in rows]
    return rows, ids


def main():
    (OUT / "connectivity_matrix").mkdir(parents=True, exist_ok=True)
    (OUT / "hit_list").mkdir(parents=True, exist_ok=True)
    combined = []
    summary = []

    for target, label, apo in TARGETS:
        source = ROOT / "results" / target
        rows, ids = residue_table(target)
        matrix = np.loadtxt(source / "connectivity.csv", delimiter=",")
        if matrix.shape != (len(ids), len(ids)):
            raise SystemExit(target + ": matrix and residue table disagree")
        if not np.allclose(matrix, matrix.T) or np.abs(np.diag(matrix)).max() > 0:
            raise SystemExit(target + ": expected a symmetric matrix with zero diagonal")

        path = OUT / "connectivity_matrix" / (target + "_connectivity.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["residue_id"] + ids)
            for name, row in zip(ids, matrix):
                writer.writerow([name] + [PRECISION % v for v in row])

        index = OUT / "connectivity_matrix" / (target + "_residue_index.csv")
        with open(index, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["matrix_index", "residue_id", "chain_id",
                             "residue_number", "residue_name"])
            for i, (r, name) in enumerate(zip(rows, ids)):
                writer.writerow([i, name, r["chain_id"], r["residue_number"],
                                 r["residue_name"]])

        hits = list(csv.DictReader(open(source / "hit_list.csv")))
        if len(hits) != 5:
            raise SystemExit(target + ": the hit list must hold exactly five residues")
        by_id = {rw["matrix_index"]: rw for rw in rows}

        # Selection and ranking answer different questions. The subset optimiser
        # chooses which five residues go together and gives every member of the
        # optimal subset the same marginal utility, so it cannot order them. The
        # rank therefore comes from the propagation score, the same quantity the
        # pipeline ranks cavities with.
        ctx = build(target)
        position = {k: i for i, k in enumerate(ctx["keys"])}
        scored = sorted(hits, key=lambda h: (-float(ctx["hop"][position[h["residue_id"]]]),
                                             int(h["matrix_index"])))
        out_rows = []
        for rank, h in enumerate(scored, start=1):
            rw = by_id[h["matrix_index"]]
            out_rows.append(dict(rank=rank, residue_id=h["residue_id"],
                                 chain_id=rw["chain_id"],
                                 residue_number=rw["residue_number"],
                                 residue_name=rw["residue_name"],
                                 propagation_score="%.6g" % ctx["hop"][position[h["residue_id"]]],
                                 subset_score=h["score"]))
        fields = ["rank", "residue_id", "chain_id", "residue_number",
                  "residue_name", "propagation_score", "subset_score"]
        with open(OUT / "hit_list" / (target + "_hit_list.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(out_rows)
        for row in out_rows:
            combined.append(dict(target=label, apo_structure=apo, **row))

        summary.append(dict(target=label, key=target, apo=apo, residues=len(ids),
                            top_five=";".join(r["residue_id"] for r in out_rows),
                            selected_unordered=";".join(sorted(h["residue_id"] for h in hits))))
        print("%-7s %-16s %4d residues  top5 %s"
              % (target, label, len(ids), ";".join(r["residue_id"] for r in out_rows)))

    fields = ["target", "apo_structure", "rank", "residue_id", "chain_id",
              "residue_number", "residue_name", "propagation_score", "subset_score"]
    with open(OUT / "hit_list" / "all_targets_hit_list.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(combined)

    (OUT / "MANIFEST.json").write_text(json.dumps(
        dict(outputs={"connectivity_matrix": "N x N per target, labelled on both axes",
                      "hit_list": "top five ranked residues per target"},
             convention=dict(value="time-averaged CTQW transfer probability between two residues",
                             symmetry="symmetric", diagonal="zero",
                             residue_id="chain:number, matching the apo structure numbering",
                             precision="10 significant digits",
                             hit_list_selection="exact fixed-cardinality subset optimisation "
                                                "inside the selected cavity",
                             hit_list_rank="descending hop-normalised propagation score; "
                                           "the subset score is equal across the five by "
                                           "construction and cannot order them",
                             hit_list_ties="residues with an equal propagation score keep "
                                           "matrix-index order"),
             targets=summary), indent=2) + "\n")
    print("\nbundle:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
