"""Residue identity, orthosteric source selection, and within-hop CTQW scores."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Sequence

import numpy as np

from scipy.sparse.csgraph import shortest_path

from scipy.stats import rankdata

from .structure import ResidueNode

DEFAULT_TIMES = np.geomspace(0.02, 20.0, 120)

CTQW_DECAY_TIME = 14.0

POCKET_TIMES = np.linspace(0.02, 200.0, 401)

@dataclass(frozen=True)
class SourceSelection:
    """Validation-independent ranking of candidate orthosteric launch residues."""

    candidate_indices: tuple[int, ...]
    concentrations: tuple[float, ...]
    selected_indices: tuple[int, ...]

def residue_id(residue: ResidueNode) -> str:
    """Return the stable ``chain:author_number[insertion]`` output identifier."""

    insertion = residue.insertion_code or ""
    return f"{residue.chain_id}:{residue.residue_number}{insertion}"

def resolve_sources(
    residues: Sequence[ResidueNode],
    source_ids: Iterable[str],
) -> tuple[int, ...]:
    """Map user-facing residue IDs to internal zero-based matrix indices.

    Source identity is a biological prior.  This function validates and resolves it;
    it does not infer catalytic residues from the structure.
    """

    lookup = {residue_id(residue): index for index, residue in enumerate(residues)}
    requested = tuple(dict.fromkeys(source_ids))
    if not requested:
        raise ValueError("at least one catalytic source residue is required")
    missing = [value for value in requested if value not in lookup]
    if missing:
        raise ValueError(f"source residues are absent from the selected structure: {missing}")
    return tuple(lookup[value] for value in requested)

def local_sequence_mask(
    residues: Sequence[ResidueNode],
    source_indices: Iterable[int],
    span: int,
) -> tuple[int, ...]:
    """Return same-chain residues inside the source-centered sequence window.

    These residues are excluded only from final candidate ranking.  They remain in
    the ENM and in quantum propagation so the local network is not physically cut.
    """

    if not isinstance(span, (int, np.integer)) or span < 0:
        raise ValueError("source exclusion span must be a non-negative integer")
    sources = tuple(source_indices)
    return tuple(
        index
        for index, residue in enumerate(residues)
        if any(
            residue.chain_id == residues[source].chain_id
            and abs(residue.residue_number - residues[source].residue_number) <= span
            for source in sources
        )
    )

def select_distal_sources(
    residues: Sequence[ResidueNode],
    connectivity: np.ndarray,
    orthosteric_indices: Iterable[int],
    *,
    local_span: int = 10,
    concentration_size: int = 5,
    source_count: int = 1,
) -> SourceSelection:
    """Select orthosteric sources whose CTQW signal is focused distally.

    For each candidate, the score is the probability mass in its strongest
    ``concentration_size`` distal residues.  The complete orthosteric set and the
    candidate's sequence-local neighborhood are excluded before sorting.  No
    allosteric labels or validation structure participate in this calculation.
    """

    residue_tuple = tuple(residues)
    matrix = np.asarray(connectivity, dtype=float)
    candidates = tuple(dict.fromkeys(int(index) for index in orthosteric_indices))
    if matrix.shape != (len(residue_tuple), len(residue_tuple)):
        raise ValueError("connectivity must match the residue count")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("connectivity must contain finite non-negative values")
    if not candidates or any(index < 0 or index >= len(residue_tuple) for index in candidates):
        raise ValueError("orthosteric indices must contain valid residues")
    if not isinstance(local_span, (int, np.integer)) or local_span < 0:
        raise ValueError("local_span must be a non-negative integer")
    if not isinstance(concentration_size, (int, np.integer)) or concentration_size < 1:
        raise ValueError("concentration_size must be a positive integer")
    if not isinstance(source_count, (int, np.integer)) or not 1 <= source_count <= len(candidates):
        raise ValueError("source_count must fit inside the orthosteric candidate set")

    candidate_set = set(candidates)
    concentrations = []
    for source in candidates:
        source_residue = residue_tuple[source]
        distal = [
            index
            for index, residue in enumerate(residue_tuple)
            if index not in candidate_set
            and not (
                residue.chain_id == source_residue.chain_id
                and abs(residue.residue_number - source_residue.residue_number)
                <= local_span
            )
        ]
        if not distal:
            raise ValueError("orthosteric and local masks leave no distal residues")
        strongest = np.sort(matrix[source, distal])[::-1][:concentration_size]
        concentrations.append(float(strongest.sum()))

    order = np.argsort(-np.asarray(concentrations), kind="stable")
    selected = tuple(candidates[int(position)] for position in order[:source_count])
    return SourceSelection(
        candidate_indices=candidates,
        concentrations=tuple(concentrations),
        selected_indices=selected,
    )

def hop_normalized_source_scores(
    connectivity: np.ndarray,
    adjacency: np.ndarray,
    source_indices: Iterable[int],
) -> tuple[np.ndarray, float]:
    """Remove graph-distance decay by ranking CTQW within each hop shell.

    The returned diagnostic is the fraction of global log-connectivity variance
    explained by hop-shell medians.  It allows routing without protein names or
    allosteric labels: a high value indicates that raw occupancy is dominated by
    distance from the orthosteric region.
    """

    matrix = np.asarray(connectivity, dtype=float)
    graph = np.asarray(adjacency, dtype=float)
    sources = tuple(dict.fromkeys(int(index) for index in source_indices))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("connectivity must be square")
    if graph.shape != matrix.shape or not np.allclose(graph, graph.T):
        raise ValueError("adjacency must be symmetric and match connectivity")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("connectivity must contain finite non-negative values")
    if not sources or any(index < 0 or index >= len(matrix) for index in sources):
        raise ValueError("source_indices must contain valid nodes")

    hops = shortest_path(graph, directed=False, unweighted=True)
    if not np.isfinite(hops).all():
        raise ValueError("hop normalization requires a connected graph")
    upper = np.triu_indices(len(matrix), 1)
    log_values = np.log10(np.maximum(matrix[upper], 1e-300))
    shell_fit = np.empty_like(log_values)
    upper_hops = hops[upper]
    for hop in np.unique(upper_hops).astype(int):
        members = upper_hops == hop
        shell_fit[members] = np.median(log_values[members])
    variance = float(np.var(log_values))
    explained = (
        0.0
        if variance <= 1e-15
        else 1.0 - float(np.var(log_values - shell_fit)) / variance
    )

    percentiles = np.zeros_like(matrix)
    for source in sources:
        for hop in np.unique(hops[source]).astype(int):
            if hop == 0:
                continue
            shell = np.flatnonzero(hops[source] == hop)
            percentiles[source, shell] = (
                rankdata(matrix[source, shell], method="average") / len(shell)
            )
    return percentiles[list(sources)].mean(axis=0), float(np.clip(explained, 0.0, 1.0))
