"""Stage 00 - resolve and fingerprint the apo and holo structures.

Reads   : data/<target>/<apo>.cif, validation_data/<target>/<holo>.cif
Writes  : 00_structures/structures.json, 00_structures/nodes.csv

Only standard amino-acid C-alpha atoms of the requested chain and author-number
range become nodes. Hetero atoms are discarded at parse time so no ligand,
solvent, ion, or cofactor can leak into a prediction. Insertion codes are
rejected because cavity membership is keyed on the author residue number.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.targets import ORDER, TARGETS
from qwallo.pipeline import residue_id
from qwallo.structure import parse_mmcif_residues
from qwallo.workspace import stage_dir, targets_from_argument, write_csv, write_json


def run(target, work):
    spec = TARGETS[target]
    apo = ROOT / "data" / target / spec["apo"]
    holo = ROOT / "validation_data" / target / spec["holo"]
    for path in (apo, holo):
        if not path.exists():
            raise SystemExit("missing structure: " + str(path))
    nodes = parse_mmcif_residues(
        apo, chains=[spec["chain"]],
        residue_ranges={spec["chain"]: spec["residue_range"]},
    )
    if any(node.insertion_code for node in nodes):
        raise SystemExit("domain contains insertion codes; cavity keys need plain numbers")
    out = stage_dir(work, target, 0, create=True)
    write_csv(out / "nodes.csv",
              ("matrix_index", "residue_id", "chain_id", "residue_number",
               "insertion_code", "residue_name", "x", "y", "z"),
              [(i, residue_id(n), n.chain_id, n.residue_number, n.insertion_code,
                n.residue_name, *[float(v) for v in n.coordinate])
               for i, n in enumerate(nodes)])
    write_json(out / "structures.json", dict(
        target=target, name=spec["name"], chain=spec["chain"],
        residue_range=list(spec["residue_range"]),
        apo=dict(file=spec["apo"], sha256=hashlib.sha256(apo.read_bytes()).hexdigest()),
        holo=dict(file=spec["holo"], partner=spec["partner"],
                  sha256=hashlib.sha256(holo.read_bytes()).hexdigest()),
        node_count=len(nodes),
        residue_numbers=[int(n.residue_number) for n in nodes],
    ))
    print(target + ": " + str(len(nodes)) + " nodes  "
          + str(nodes[0].residue_number) + "-" + str(nodes[-1].residue_number))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *ORDER))
    parser.add_argument("--work", default=str(ROOT / "work"))
    args = parser.parse_args()
    for target in targets_from_argument(args.target, ORDER):
        run(target, args.work)


if __name__ == "__main__":
    main()
