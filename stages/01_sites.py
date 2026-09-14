"""Stage 01 - define the active site (seed) and the allosteric site (truth).

Reads   : 00_structures/nodes.csv, validation_data/<target>/<holo>.cif
Writes  : 01_sites/<target>_nodes.csv, 01_sites/sites.json

Both sites use one criterion: a residue belongs to a site when it has a heavy
atom within 4.5 A of any heavy atom of that site's **binding partner** - the
bound entity occupying the site, whether a small molecule, metal ion,
nucleotide, nucleic-acid chain, or peptide. The two definitions therefore rest
on the same scale.

The active site is supplied as an annotation prior in config/targets.py. This
stage records it, and additionally reports how it compares with the partner
contacts derived from the same holo file, as a diagnostic only.

The node table carries ``site_class`` in {orthosteric, allosteric, shared,
unlabelled}; those four strings are a machine-readable contract and are used
literally by stages 05 and 06.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER, TARGETS
from qwallo.constants import CONTACT_RADIUS
from qwallo.sites import partner_contacts, site_classes
from qwallo.workspace import (read_csv, stage_dir, targets_from_argument,
                             write_csv, write_json)


def run(target, work):
    spec = TARGETS[target]
    rows = read_csv(stage_dir(work, target, 0) / "nodes.csv")
    keys = [row["residue_id"] for row in rows]
    seed_ids = [spec["chain"] + ":" + str(number) for number in spec["orthosteric"]]
    missing = [key for key in seed_ids if key not in set(keys)]
    if missing:
        raise SystemExit("orthosteric residues absent from the domain: " + str(missing))
    holo = ROOT / "validation_data" / target / spec["holo"]
    truth_ids = partner_contacts(holo, spec["partner"], chain=spec["chain"],
                                 radius=CONTACT_RADIUS)
    classes = site_classes(keys, seed_ids, truth_ids)
    out = stage_dir(work, target, 1, create=True)
    write_csv(out / (target + "_nodes.csv"),
              ("matrix_index", "resnum", "icode", "residue_id", "residue_name",
               "site_class", "x", "y", "z"),
              [(row["matrix_index"], row["residue_number"], row["insertion_code"],
                row["residue_id"], row["residue_name"], site,
                row["x"], row["y"], row["z"])
               for row, site in zip(rows, classes)])
    inside = sorted(key for key in truth_ids if key in set(keys))
    write_json(out / "sites.json", dict(
        target=target, contact_radius_angstrom=CONTACT_RADIUS,
        active_site=dict(source="annotation prior (config/targets.py)",
                         residues=seed_ids, count=len(seed_ids)),
        allosteric_site=dict(source="holo binding-partner heavy-atom contact",
                             holo=spec["holo"], partner=spec["partner"],
                             residues=sorted(truth_ids), count=len(truth_ids),
                             inside_domain=inside, inside_domain_count=len(inside)),
        site_class_counts={name: classes.count(name) for name in
                           ("orthosteric", "allosteric", "shared", "unlabelled")},
    ))
    counts = {name: classes.count(name) for name in ("orthosteric", "allosteric", "shared")}
    print(target + ": seeds " + str(counts["orthosteric"] + counts["shared"])
          + "  allosteric " + str(counts["allosteric"] + counts["shared"])
          + "  shared " + str(counts["shared"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work)


if __name__ == "__main__":
    main()
