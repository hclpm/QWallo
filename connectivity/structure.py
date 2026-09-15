"""Convert an mmCIF structure into the residue graph used by every predictor.

Only standard amino-acid C-alpha atoms become graph nodes.  Hetero atoms are
intentionally discarded here so ligands, solvent, ions, and cofactors cannot leak
validation information into prediction.
"""

from dataclasses import dataclass

from pathlib import Path

from typing import Collection, Mapping, Optional, Sequence, Tuple

import numpy as np

from Bio.PDB import MMCIFParser, is_aa

@dataclass(frozen=True)
class ResidueNode:
    """One coarse-grained residue and the PDB identity needed for output mapping.

    The list position is the matrix index.  ``residue_number`` is the author residue
    number from the structure and must not be treated as an array index.
    """

    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    coordinate: np.ndarray

def parse_mmcif_residues(
    path: Path,
    chains: Optional[Collection[str]] = None,
    residue_ranges: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> list[ResidueNode]:
    """Read selected standard amino-acid C-alpha nodes from the first model.

    Args:
        path: Input mmCIF file.
        chains: Optional set of chain identifiers.  ``None`` keeps every chain.
        residue_ranges: Optional inclusive author-number range for each chain.

    Returns:
        Residues in the deterministic model/chain/file order produced by Biopython.
    """

    selected_chains = set(chains) if chains is not None else None
    structure = MMCIFParser(QUIET=True).get_structure(Path(path).stem, str(path))
    # NMR and ensemble files can contain several models.  Mixing them would create
    # duplicate residue identities, so the challenge pipeline consistently uses one.
    model = next(structure.get_models())
    nodes = []

    for chain in model:
        if selected_chains is not None and chain.id not in selected_chains:
            continue
        selected_range = residue_ranges.get(chain.id) if residue_ranges else None
        for residue in chain:
            hetero_flag, residue_number, insertion_code = residue.id
            # A blank hetero flag excludes water, ligands, ions, and modified groups.
            # ``standard=True`` also prevents nonstandard PTMs from becoming nodes.
            if hetero_flag != " " or not is_aa(residue, standard=True) or "CA" not in residue:
                continue
            if selected_range is not None:
                start, end = selected_range
                if not start <= residue_number <= end:
                    continue
            nodes.append(
                ResidueNode(
                    chain_id=chain.id,
                    residue_number=int(residue_number),
                    insertion_code=insertion_code.strip(),
                    residue_name=residue.resname,
                    coordinate=np.asarray(residue["CA"].coord, dtype=float).copy(),
                )
            )

    if not nodes:
        raise ValueError("No standard amino-acid C-alpha residues matched the selection")
    return nodes

def build_uniform_contact_network(
    residues: Sequence[ResidueNode],
    cutoff: float = 7.8,
) -> np.ndarray:
    """Construct the binary C-alpha elastic network adjacency matrix.

    Two distinct residues are joined by a unit spring when their C-alpha distance is
    at most ``cutoff``.  The result is symmetric, has a zero diagonal, and contains
    no isolated nodes.
    """

    if len(residues) < 2:
        raise ValueError("At least two residues are required")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")

    coordinates = np.stack([residue.coordinate for residue in residues])
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=-1,
    )
    # This is the Gaussian Network Model contact rule A_ij = 1[d_ij <= cutoff].
    adjacency = ((distances <= cutoff) & (distances > 0.0)).astype(float)
    np.fill_diagonal(adjacency, 0.0)

    isolated = np.flatnonzero(adjacency.sum(axis=1) <= 0.0)
    if isolated.size:
        raise ValueError(
            f"Elastic network contains isolated residue indices: {isolated.tolist()}"
        )
    return adjacency
