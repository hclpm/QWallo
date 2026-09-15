"""Pack output 1 into a single workbook, one sheet per target.

The per-target CSV matrices stay in the repository; this writes the same numbers
into one file so the submission carries fewer attachments. Sheet layout mirrors
the CSV exactly: row 1 and column A hold the residue id, so the cell at
(row i, column j) is the connectivity between those two residues.

A residue_index sheet carries the matrix order with residue names, and a README
sheet states the convention, so the workbook is self-contained.
"""

import csv
import json
from pathlib import Path

import numpy as np
import xlsxwriter

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission"
OUT = SRC / "QWallo_outputs.xlsx"
TARGETS = [("kras", "KRAS_G12C"), ("bcr", "BCR-ABL1"),
           ("myosin", "Cardiac_myosin"), ("cmyc", "c-Myc_Max")]


def main():
    manifest = json.loads((SRC / "MANIFEST.json").read_text())
    book = xlsxwriter.Workbook(str(OUT), {"constant_memory": True, "strings_to_numbers": False})
    head = book.add_format({"bold": True, "bg_color": "#EEEEEE", "border": 1})
    label = book.add_format({"bold": True, "bg_color": "#F7F7F7", "border": 1})
    number = book.add_format({"num_format": "0.000000000"})

    readme = book.add_worksheet("README")
    readme.set_column(0, 0, 26); readme.set_column(1, 1, 96)
    notes = [("Outputs", "1. Connectivity Matrix and 2. Hit List "
                          "(Cleveland Clinic challenge statement, section 5)"),
             ("Matrix sheets", "one per target: " + ", ".join(s for _, s in TARGETS)),
             ("Hit list sheet", "hit_list holds the ranked top five for all four targets"),
             ("Layout", "row 1 and column A hold the residue id; cell (i, j) is the "
                        "connectivity between residue i and residue j"),
             ("Value", manifest["convention"]["value"]),
             ("Symmetry", manifest["convention"]["symmetry"]),
             ("Diagonal", manifest["convention"]["diagonal"]),
             ("Residue id", manifest["convention"]["residue_id"]),
             ("Precision", manifest["convention"]["precision"]),
             ("Residue order", "see the residue_index sheet; it matches the matrix axes"),
             ("Hit list selection", manifest["convention"]["hit_list_selection"]),
             ("Hit list rank", manifest["convention"]["hit_list_rank"]),
             ("Hit list ties", manifest["convention"]["hit_list_ties"]),
             ("Source", "github.com/hclpm/QWallo, submission/connectivity_matrix/")]
    readme.write_row(0, 0, ["Field", "Value"], head)
    for r, (k, v) in enumerate(notes, start=1):
        readme.write(r, 0, k, label); readme.write(r, 1, v)

    index_rows = []
    for key, sheet_name in TARGETS:
        rows = list(csv.reader(open(SRC / "connectivity_matrix" / (key + "_connectivity.csv"))))
        ids = rows[0][1:]
        matrix = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
        if [r[0] for r in rows[1:]] != ids:
            raise SystemExit(key + ": the two axes disagree")

        sheet = book.add_worksheet(sheet_name)
        sheet.freeze_panes(1, 1)
        sheet.set_column(0, 0, 12)
        sheet.write(0, 0, "residue_id", head)
        sheet.write_row(0, 1, ids, head)
        for i, (name, row) in enumerate(zip(ids, matrix), start=1):
            sheet.write(i, 0, name, label)
            sheet.write_row(i, 1, row.tolist(), number)
        print("%-16s %d x %d" % (sheet_name, len(ids), len(ids)), flush=True)

        for r in csv.DictReader(open(SRC / "connectivity_matrix" / (key + "_residue_index.csv"))):
            index_rows.append([sheet_name, int(r["matrix_index"]), r["residue_id"],
                               r["chain_id"], int(r["residue_number"]), r["residue_name"]])

    hits = list(csv.DictReader(open(SRC / "hit_list" / "all_targets_hit_list.csv")))
    sheet = book.add_worksheet("hit_list")
    fields = list(hits[0])
    sheet.write_row(0, 0, fields, head)
    for r, row in enumerate(hits, start=1):
        for cel, field in enumerate(fields):
            value = row[field]
            if field in ("rank", "residue_number"):
                sheet.write_number(r, cel, int(value))
            elif field in ("propagation_score", "subset_score"):
                sheet.write_number(r, cel, float(value))
            else:
                sheet.write(r, cel, value)
    sheet.set_column(0, 1, 16); sheet.set_column(3, 6, 14)

    index = book.add_worksheet("residue_index")
    index.write_row(0, 0, ["sheet", "matrix_index", "residue_id", "chain_id",
                           "residue_number", "residue_name"], head)
    for r, row in enumerate(index_rows, start=1):
        index.write_row(r, 0, row)
    index.set_column(0, 0, 16); index.set_column(2, 2, 12)
    book.close()
    print("\nworkbook", OUT, OUT.stat().st_size, "bytes |", len(index_rows), "residues indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
