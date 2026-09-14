"""Site definition from binding-partner heavy-atom contacts.

A *binding partner* is the bound entity that occupies a site in the reference
structure: a small molecule, metal ion, nucleotide, nucleic-acid chain, or
peptide. The term is used instead of "ligand" because in the PDB data model a
ligand is a non-polymer chemical component, which excludes DNA, RNA, and
peptide chains that define sites on many targets.

Both sites share one criterion, so they rest on the same scale:

    site residues = residues with a heavy atom within CONTACT_RADIUS of any
                    heavy atom of the binding partner

The **active site** (propagation seed) is supplied as an annotation prior in
``config/targets.py``; this module can also derive it from an apo binding
partner as a cross-check. The **allosteric site** (ground truth) is derived
here from the holo structure and is never read by a prediction stage.
"""

from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, is_aa

from .constants import CONTACT_RADIUS


def partner_contacts(path, partner, chain="A", radius=CONTACT_RADIUS):
    """Return ``{chain:number}`` residues contacting the named binding partner.

    Hydrogens and deuteriums are excluded on both sides, so the criterion is a
    heavy-atom contact. Only standard amino acids of ``chain`` are returned.
    """
    model = next(MMCIFParser(QUIET=True).get_structure("ref", str(path)).get_models())
    partner_atoms = np.asarray([
        atom.coord
        for residue in model.get_residues()
        if residue.resname.strip() == partner
        for atom in residue
        if str(atom.element).upper() not in {"H", "D"}
    ])
    if not len(partner_atoms):
        raise ValueError("binding partner " + partner + " is absent from " + str(path))
    contacts = set()
    for residue in model[chain]:
        flag, number, insertion = residue.id
        if flag != " " or not is_aa(residue, standard=True):
            continue
        atoms = np.asarray([
            atom.coord for atom in residue
            if str(atom.element).upper() not in {"H", "D"}
        ])
        if len(atoms) and np.any(
            np.linalg.norm(atoms[:, None] - partner_atoms[None, :], axis=2) <= radius
        ):
            contacts.add(chain + ":" + str(number) + insertion.strip())
    return contacts


def site_classes(residue_ids, seed_ids, truth_ids):
    """Label every node ``orthosteric`` / ``allosteric`` / ``shared`` / ``unlabelled``.

    A residue in both sets is ``shared``: it contacts both binding partners.
    Downstream scoring excludes seeds, so ``shared`` residues are excluded too
    unless a caller overrides that policy.
    """
    seed, truth = set(seed_ids), set(truth_ids)
    out = []
    for key in residue_ids:
        in_seed, in_truth = key in seed, key in truth
        if in_seed and in_truth:
            out.append("shared")
        elif in_seed:
            out.append("orthosteric")
        elif in_truth:
            out.append("allosteric")
        else:
            out.append("unlabelled")
    return out
