"""Evaluate saved predictions against independent holo ligand contacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, is_aa
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
TARGETS = {
    "kras": ("6OIM.cif", "MOV"),
    "bcr": ("5MO4.cif", "AY7"),
    "myosin": ("9YR7.cif", "XB2"),
    "mapk14": ("3NEW.cif", "3NE"),
    "ptpn1": ("1T48.cif", "BB3"),
    "pdk2": ("2BU2.cif", "TF1"),
    "ptpn11": ("5EHP.cif", "5OA"),
}


def ligand_contacts(path: Path, ligand: str, radius: float = 4.5) -> set[str]:
    model = next(MMCIFParser(QUIET=True).get_structure("holo", path).get_models())
    ligand_atoms = np.asarray([
        atom.coord
        for residue in model.get_residues()
        if residue.resname.strip() == ligand
        for atom in residue
        if str(atom.element).upper() not in {"H", "D"}
    ])
    if not len(ligand_atoms):
        raise ValueError(f"ligand {ligand} is absent from {path}")
    contacts = set()
    for residue in model["A"]:
        flag, number, insertion = residue.id
        if flag != " " or not is_aa(residue, standard=True):
            continue
        atoms = np.asarray([
            atom.coord for atom in residue
            if str(atom.element).upper() not in {"H", "D"}
        ])
        if len(atoms) and np.any(
            np.linalg.norm(atoms[:, None] - ligand_atoms[None, :], axis=2) <= radius
        ):
            contacts.add(f"A:{number}{insertion.strip()}")
    return contacts


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("evaluation requires positive and negative residues")
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(labels) - 1]
    true_positives = np.cumsum(ordered_labels)[ends]
    average_precision = float(np.sum(
        np.diff(np.r_[0, true_positives]) / positives * true_positives / (ends + 1)
    ))
    ranks = rankdata(scores, method="average")
    negatives = len(labels) - positives
    auroc = (
        float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    return {
        "residue_count": len(labels),
        "positive_count": positives,
        "random_auprc": positives / len(labels),
        "auprc": average_precision,
        "auroc": float(auroc),
        "hits_at_5": int(labels[order[:5]].sum()),
    }


def validate(target: str, results_root=ROOT / "results") -> dict:
    holo, ligand = TARGETS[target]
    output = Path(results_root) / target
    with (output / "residue_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    residue_ids = [
        f"{row['chain_id']}:{row['residue_number']}{row['insertion_code']}"
        for row in rows
    ]
    scores = np.asarray([float(row["quantum_score"]) for row in rows])
    truth = ligand_contacts(ROOT / "validation_data" / target / holo, ligand)
    labels = np.asarray([residue in truth for residue in residue_ids], dtype=int)
    result = {
        "target": target,
        "holo_structure": holo,
        "ligand": ligand,
        "contact_radius_angstrom": 4.5,
        "ground_truth_residues": sorted(truth),
        "metrics": metrics(labels, scores),
    }
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *TARGETS))
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for target in TARGETS if args.target == "all" else (args.target,):
        result = validate(target, args.results_root)["metrics"]
        print(
            f"{target:7s} top5={result['hits_at_5']}/5 "
            f"AUPRC={result['auprc']:.4f} random={result['random_auprc']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
