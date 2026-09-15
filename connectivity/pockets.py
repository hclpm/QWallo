"""ENM pocket proposals and orthosteric-to-cavity CTQW ranking.

fpocket supplies geometry-only cavity observations across an apo ENM ensemble.
Within-hop CTQW consensus then selects a distal cavity family without using a
holo structure or known allosteric residue.
"""

from dataclasses import dataclass

from copy import deepcopy

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import re

import shutil

import subprocess

import tempfile

import numpy as np

from Bio.PDB import MMCIFParser, PDBIO, Select, is_aa

from scipy.sparse.csgraph import shortest_path

from scipy.stats import rankdata

from .anm import _low_anm_modes, anm_fluctuation_connectivity, build_anm_hessian

from .structure import ResidueNode

@dataclass(frozen=True)
class Pocket:
    number: int
    geometry_score: float
    residue_numbers: tuple[int, ...]
    volume: float = 0.0
    persistence: float = 1.0
    opening: float = 0.0
    center: tuple[float, float, float] | None = None
    residue_persistence: tuple[tuple[int, float], ...] = ()
    observations: tuple[tuple[int, ...], ...] = ()
    void_support: tuple[tuple[int, float], ...] = ()

@dataclass(frozen=True)
class PocketRanking:
    selected_pocket: tuple[int, ...]
    candidate_pockets: tuple[tuple[int, ...], ...]
    candidate_geometry_scores: tuple[float, ...]
    candidate_quantum_scores: tuple[float, ...]
    ranking_method: str = "ctqw"

def orthosteric_region_pocket_scores(
    raw_source_scores: np.ndarray,
    hop_source_scores: np.ndarray,
    candidate_pockets,
    *,
    hop_dominated: bool,
    top_k: int = 5,
) -> np.ndarray:
    """Score cavities from the complete orthosteric region, not one residue.

    Raw transport is averaged as an incoherent mixture of source states. When
    graph distance dominates the CTQW matrix, each source first ranks candidate
    cavities by within-hop transport and those ranks are averaged.
    """

    raw = np.asarray(raw_source_scores, dtype=float)
    hop = np.asarray(hop_source_scores, dtype=float)
    pockets = tuple(
        tuple(dict.fromkeys(int(node) for node in group))
        for group in candidate_pockets
    )
    if raw.ndim != 2 or hop.shape != raw.shape or raw.shape[0] < 1:
        raise ValueError("source score arrays must be equal-sized non-empty matrices")
    if not np.isfinite(raw).all() or not np.isfinite(hop).all():
        raise ValueError("source score arrays must contain finite values")
    if not pockets or any(not group for group in pockets):
        raise ValueError("candidate_pockets must contain non-empty groups")
    if top_k < 1 or any(
        node < 0 or node >= raw.shape[1] for group in pockets for node in group
    ):
        raise ValueError("top_k and candidate pocket indices must be valid")

    def strongest_mean(values, group):
        count = min(top_k, len(group))
        return float(np.sort(values[list(group)])[-count:].mean())

    if not hop_dominated:
        mixed = raw.mean(axis=0)
        return np.asarray([strongest_mean(mixed, group) for group in pockets])

    source_by_pocket = np.asarray(
        [[strongest_mean(row, group) for group in pockets] for row in hop]
    )
    # Within-hop values are already source-calibrated percentiles. Averaging the
    # values retains evidence strength; ranking them a second time discards it.
    return source_by_pocket.mean(axis=0)

def select_mechanically_supported_cavity_field(
    centers: np.ndarray,
    candidate_pockets,
    *,
    transport_scores: np.ndarray,
    mechanical_scores: np.ndarray,
    radius: float = 8.0,
    mechanical_tolerance: float = 0.95,
    direct_index: int | None = None,
    direct_pocket=None,
) -> tuple[tuple[int, ...], bool]:
    """Resolve a fragmented cavity using local recurrence and ANM support.

    CTQW first selects the neighborhood with the highest mean transport. A
    recurrent lining field replaces a singleton winner only when its strongest
    ANM response is comparable to the singleton response. Neighborhoods are
    centered independently, so contact chains cannot merge the protein surface.
    """

    points = np.asarray(centers, dtype=float)
    pockets = tuple(
        tuple(dict.fromkeys(int(node) for node in group))
        for group in candidate_pockets
    )
    transport = np.asarray(transport_scores, dtype=float)
    mechanical = np.asarray(mechanical_scores, dtype=float)
    count = len(pockets)
    if (
        points.shape != (count, 3)
        or transport.shape != (count,)
        or mechanical.shape != (count,)
        or count == 0
        or any(not group for group in pockets)
        or not np.isfinite(points).all()
        or not np.isfinite(transport).all()
        or not np.isfinite(mechanical).all()
        or radius <= 0.0
        or not 0.0 < mechanical_tolerance <= 1.0
    ):
        raise ValueError("valid cavity centers, scores, radius, and tolerance are required")

    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    neighborhoods = []
    seen = set()
    for anchor in range(count):
        members = tuple(int(index) for index in np.flatnonzero(distances[anchor] <= radius))
        if members in seen:
            continue
        seen.add(members)
        nodes = tuple(sorted(set().union(*(set(pockets[index]) for index in members))))
        support = np.asarray(
            [
                sum(transport[index] for index in members if node in pockets[index])
                for node in nodes
            ]
        )
        strongest_count = min(5, len(support))
        neighborhoods.append(
            {
                "members": members,
                "nodes": nodes,
                "transport": float(transport[list(members)].mean()),
                "recurrence": float(
                    np.sort(support)[-strongest_count:].mean() / np.sqrt(len(members))
                ),
                "mechanical": float(mechanical[list(members)].max()),
            }
        )

    direct_index = int(np.argmax(transport)) if direct_index is None else int(direct_index)
    if direct_index < 0 or direct_index >= count:
        raise ValueError("direct_index must identify a candidate pocket")
    direct_members = tuple(
        int(index) for index in np.flatnonzero(distances[direct_index] <= radius)
    )
    direct = {
        "members": direct_members,
        "nodes": (
            pockets[direct_index]
            if direct_pocket is None
            else tuple(sorted(dict.fromkeys(int(node) for node in direct_pocket)))
        ),
        "transport": float(transport[direct_index]),
        "mechanical": float(mechanical[direct_index]),
    }
    recurrent = max(neighborhoods, key=lambda row: row["recurrence"])
    use_recurrence = (
        len(direct["members"]) == 1
        and direct["mechanical"] > 0.0
        and recurrent["mechanical"]
        >= mechanical_tolerance * direct["mechanical"]
    )
    selected = recurrent if use_recurrence else direct
    return selected["nodes"], use_recurrence


def chain_ranges(chain, start=None, end=None):
    """Normalise a chain specification to {chain_id: (start, end)}.

    A plain identifier with a start and an end is the one-entry case and keeps
    the single-chain behaviour unchanged. A mapping selects several chains, each
    with its own inclusive author-number window, which is what a multimer target
    needs. Cavity lining is keyed on the residue number alone, so the selected
    chains must not share numbers; the caller checks that.
    """
    if isinstance(chain, dict):
        return {str(key): (int(a), int(b)) for key, (a, b) in chain.items()}
    return {str(chain): (int(start), int(end))}


class _DomainSelect(Select):
    def __init__(self, ranges):
        self.ranges = dict(ranges)

    def accept_chain(self, chain):
        return chain.id in self.ranges

    def accept_residue(self, residue):
        flag, number, _ = residue.id
        window = self.ranges.get(residue.get_parent().id)
        if window is None:
            return False
        return flag == " " and window[0] <= number <= window[1] and is_aa(
            residue, standard=True
        )

def matched_spatial_null_pockets(
    residues,
    *,
    pocket_indices,
    source_indices,
    distance_tolerance: float = 4.0,
) -> tuple[tuple[int, ...], ...]:
    """Build same-size 3D patches at a matched distance from the source region."""

    residue_tuple = tuple(residues)
    pocket = tuple(dict.fromkeys(int(index) for index in pocket_indices))
    sources = tuple(dict.fromkeys(int(index) for index in source_indices))
    if (
        not residue_tuple
        or not pocket
        or not sources
        or distance_tolerance <= 0.0
        or any(index < 0 or index >= len(residue_tuple) for index in pocket + sources)
        or set(pocket).intersection(sources)
    ):
        raise ValueError("valid distal pocket, sources, and distance tolerance are required")

    coordinates = np.stack([residue.coordinate for residue in residue_tuple])
    source_coordinates = coordinates[list(sources)]
    pocket_center = coordinates[list(pocket)].mean(axis=0)
    target_distance = float(
        np.min(np.linalg.norm(source_coordinates - pocket_center, axis=1))
    )
    center_distances = np.min(
        np.linalg.norm(coordinates[:, None, :] - source_coordinates[None, :, :], axis=2),
        axis=1,
    )
    centers = np.flatnonzero(np.abs(center_distances - target_distance) <= distance_tolerance)

    pocket_set = set(pocket)
    maximum_overlap = max(3, len(pocket) // 5)
    nulls = []
    seen = set()
    for center in centers:
        patch = tuple(
            sorted(
                int(index)
                for index in np.argsort(
                    np.linalg.norm(coordinates - coordinates[center], axis=1)
                )[: len(pocket)]
            )
        )
        if (
            set(patch).intersection(sources)
            or len(set(patch).intersection(pocket_set)) > maximum_overlap
            or patch in seen
        ):
            continue
        seen.add(patch)
        nulls.append(patch)
    if not nulls:
        raise ValueError("no size- and source-distance-matched spatial null was found")
    return tuple(nulls)

def enm_mode_displacements(
    residues,
    *,
    cutoff: float = 12.0,
    mode_count: int = 2,
    amplitudes=(1.0, 2.0),
) -> tuple[np.ndarray, ...]:
    """Return deterministic +/- directional ENM modes at requested RMS amplitudes."""

    residue_tuple = tuple(residues)
    amplitude_tuple = tuple(float(value) for value in amplitudes)
    if mode_count < 1 or not amplitude_tuple or any(value <= 0.0 for value in amplitude_tuple):
        raise ValueError("mode_count and amplitudes must be positive")
    hessian, _ = build_anm_hessian(residue_tuple, cutoff=cutoff)
    _, modes = _low_anm_modes(hessian, mode_count)
    if modes.shape[1] < mode_count:
        raise ValueError("ENM contains fewer non-rigid modes than requested")

    results = []
    for amplitude in amplitude_tuple:
        for mode_index in range(mode_count):
            vectors = modes[:, mode_index].reshape(len(residue_tuple), 3)
            rms = float(np.sqrt(np.mean(np.sum(vectors**2, axis=1))))
            scaled = vectors * (amplitude / max(rms, 1e-15))
            results.extend((scaled, -scaled))
    return tuple(results)

def parse_fpocket_output(output_dir: Path, chain) -> tuple[Pocket, ...]:
    """Read geometry score and lining residues from fpocket PDB outputs.

    ``chain`` is one identifier or a collection of them.
    """

    wanted = {str(chain)} if isinstance(chain, str) else {str(c) for c in chain}

    parsed = []
    paths = sorted(
        (Path(output_dir) / "pockets").glob("pocket*_atm.pdb"),
        key=lambda path: int(re.search(r"pocket(\d+)", path.name).group(1)),
    )
    for path in paths:
        pocket_number = int(re.search(r"pocket(\d+)", path.name).group(1))
        text = path.read_text()
        score_match = re.search(r"Pocket Score\s+:\s+([0-9.]+)", text)
        if score_match is None:
            continue
        volume_match = re.search(
            r"Pocket volume \(Monte Carlo\)\s+:\s+([0-9.]+)", text
        )
        residue_atoms = {}
        for line in text.splitlines():
            if not line.startswith(("ATOM", "HETATM")) or line[21].strip() not in wanted:
                continue
            residue_number = int(line[22:26])
            try:
                coordinate = np.asarray(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            except ValueError:
                fields = line.split()
                coordinate = np.asarray(tuple(float(value) for value in fields[6:9]))
            residue_atoms.setdefault(residue_number, []).append(coordinate)
        residues = set(residue_atoms)
        if residues:
            vertex_path = path.with_name(f"pocket{pocket_number}_vert.pqr")
            center = None
            void_support = ()
            if vertex_path.exists():
                spheres, sphere_radii = _parse_alpha_sphere_geometry(vertex_path)
                center = tuple(float(value) for value in spheres.mean(axis=0))
                support = []
                for residue_number, atoms in residue_atoms.items():
                    distances = np.linalg.norm(
                        spheres[:, None, :] - np.asarray(atoms)[None, :, :], axis=2
                    ).min(axis=1)
                    support.append(
                        (
                            residue_number,
                            # An alpha sphere is tangent to its lining atoms.  Its
                            # PQR radius therefore matters: distance from the
                            # center alone would incorrectly reward atoms inside
                            # the void rather than atoms on its boundary.
                            float(
                                np.mean(
                                    np.exp(
                                        -0.5
                                        * ((distances - sphere_radii) / 0.25) ** 2
                                    )
                                )
                            ),
                        )
                    )
                void_support = tuple(sorted(support))
            parsed.append(
                Pocket(
                    pocket_number,
                    float(score_match.group(1)),
                    tuple(sorted(residues)),
                    float(volume_match.group(1)) if volume_match else 0.0,
                    center=center,
                    void_support=void_support,
                )
            )
    return tuple(parsed)

def _parse_alpha_sphere_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read alpha-sphere centers and radii from an fpocket PQR file."""

    points = []
    radii = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                point = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                fields = line.split()
                point = tuple(float(value) for value in fields[6:9])
            points.append(point)
            radii.append(float(line.split()[-1]))
    if not points:
        raise ValueError(f"no alpha-sphere coordinates found in {path}")
    return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)

def merge_stable_pocket_families(
    scout_snapshots,
    refinement_snapshots,
    *,
    stability_threshold: float = 0.5,
    family_overlap: float = 0.5,
    center_tolerance: float = 6.0,
) -> tuple[Pocket, ...]:
    """Merge cavities reproduced at two amplitudes across independent ENM modes.

    Each scout snapshot and refinement snapshot must describe the same mode and
    direction. A cavity first has to reproduce at the second amplitude; stable
    observations are then grouped by lining overlap. This retains rare cryptic
    cavities without promoting a one-off fpocket detection.
    """

    scouts = tuple(tuple(snapshot) for snapshot in scout_snapshots)
    refinements = tuple(tuple(snapshot) for snapshot in refinement_snapshots)
    if not scouts or len(scouts) != len(refinements):
        raise ValueError("paired scout and refinement snapshots are required")
    if not 0.0 < stability_threshold <= 1.0 or not 0.0 < family_overlap <= 1.0:
        raise ValueError("pocket overlap thresholds must be in (0, 1]")
    if center_tolerance <= 0.0:
        raise ValueError("center_tolerance must be positive")

    stable = []
    for event, (scout, refinement) in enumerate(zip(scouts, refinements)):
        for pocket in scout:
            numbers = set(pocket.residue_numbers)
            matches = []
            for other in refinement:
                other_numbers = set(other.residue_numbers)
                union = numbers | other_numbers
                matches.append((len(numbers & other_numbers) / len(union), other))
            stability, matched = max(matches, default=(0.0, None), key=lambda item: item[0])
            if stability >= stability_threshold:
                center = pocket.center
                if center is None and matched is not None:
                    center = matched.center
                stable.append({"event": event, "pocket": pocket, "center": center})

    families = []
    for row in sorted(
        stable,
        key=lambda item: (-len(item["pocket"].residue_numbers), item["event"]),
    ):
        numbers = set(row["pocket"].residue_numbers)
        family = None
        for candidate in families:
            anchor = candidate["anchor"]
            jaccard = len(numbers & anchor) / len(numbers | anchor)
            containment = len(numbers & anchor) / min(len(numbers), len(anchor))
            centers_close = (
                row["center"] is not None
                and candidate["center"] is not None
                and np.linalg.norm(np.asarray(row["center"]) - candidate["center"])
                <= center_tolerance
            )
            if jaccard >= family_overlap or (containment >= family_overlap and centers_close):
                family = candidate
                break
        if family is None:
            families.append(
                {
                    "anchor": numbers,
                    "center": None if row["center"] is None else np.asarray(row["center"]),
                    "members": [row],
                }
            )
        else:
            family["members"].append(row)

    merged = []
    event_count = len(scouts)
    for number, family in enumerate(families, start=1):
        members = family["members"]
        events = {member["event"] for member in members}
        residue_events = {}
        residue_void_support = {}
        for member in members:
            for residue in member["pocket"].residue_numbers:
                residue_events.setdefault(residue, set()).add(member["event"])
            for residue, support in member["pocket"].void_support:
                residue_void_support.setdefault(residue, []).append(support)
        residue_numbers = tuple(sorted(residue_events))
        centers = [member["center"] for member in members if member["center"] is not None]
        center = tuple(np.mean(centers, axis=0)) if centers else None
        merged.append(
            Pocket(
                number=number,
                geometry_score=float(
                    np.mean([member["pocket"].geometry_score for member in members])
                ),
                residue_numbers=residue_numbers,
                volume=max(member["pocket"].volume for member in members),
                persistence=len(events) / event_count,
                center=center,
                residue_persistence=tuple(
                    (residue, len(events_seen) / len(events))
                    for residue, events_seen in sorted(residue_events.items())
                ),
                observations=tuple(
                    member["pocket"].residue_numbers for member in members
                ),
                void_support=tuple(
                    (residue, float(np.mean(values)))
                    for residue, values in sorted(residue_void_support.items())
                ),
            )
        )
    return tuple(merged)

def _run_fpocket_pdb(
    pdb_path: Path,
    chain,
    executable: str,
) -> tuple[Pocket, ...]:
    """Run fpocket on one prepared PDB snapshot."""

    completed = subprocess.run(
        [executable, "-f", str(pdb_path)],
        cwd=pdb_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fpocket failed: {completed.stderr.strip()}")
    return parse_fpocket_output(
        pdb_path.with_name(f"{pdb_path.stem}_out"),
        chain,
    )

def _domain_residues(structure, ranges):
    """Return model residues in the same order used for ENM displacement rows.

    Chains are visited in sorted identifier order so the row order of a
    multi-chain target is deterministic and independent of file order.
    """

    model = next(structure.get_models())
    selected = []
    for chain_id in sorted(ranges):
        start, end = ranges[chain_id]
        for residue in model[chain_id]:
            flag, number, insertion = residue.id
            if (
                flag == " "
                and start <= number <= end
                and is_aa(residue, standard=True)
                and "CA" in residue
            ):
                selected.append(residue)
    return selected

def _sample_enm_pockets(
    mmcif: Path,
    chain,
    start=None,
    end=None,
    *,
    mode_count: int,
    amplitudes,
    anm_cutoff: float,
    executable: str,
    detector,
    workers: int = 1,
):
    """Return residue nodes and fpocket results for the reference plus ENM snapshots."""

    if workers < 1:
        raise ValueError("workers must be positive")
    command = shutil.which(executable)
    if detector is None and command is None:
        raise RuntimeError("fpocket is required; install it with `brew install fpocket`")
    ranges = chain_ranges(chain, start, end)
    structure = MMCIFParser(QUIET=True).get_structure(Path(mmcif).stem, str(mmcif))
    selected = _domain_residues(structure, ranges)
    if not selected:
        raise ValueError("no standard amino-acid residues matched the pocket domain")
    if len({int(r.id[1]) for r in selected}) != len(selected):
        raise ValueError(
            "selected chains share residue numbers; cavity lining keyed on the "
            "number alone would be ambiguous"
        )
    nodes = tuple(
        ResidueNode(
            residue.get_parent().id,
            int(residue.id[1]),
            residue.id[2].strip(),
            residue.resname,
            np.asarray(residue["CA"].coord, dtype=float).copy(),
        )
        for residue in selected
    )
    displacements = enm_mode_displacements(
        nodes,
        cutoff=anm_cutoff,
        mode_count=mode_count,
        amplitudes=amplitudes,
    )

    with tempfile.TemporaryDirectory(prefix="qai-enm-pockets-") as temporary:
        directory = Path(temporary)
        pdb_paths = []
        all_displacements = (np.zeros((len(nodes), 3)),) + displacements
        for snapshot_index, displacement in enumerate(all_displacements):
            deformed = deepcopy(structure)
            for residue, shift in zip(
                _domain_residues(deformed, ranges),
                displacement,
            ):
                for atom in residue:
                    atom.coord = np.asarray(atom.coord, dtype=float) + shift
            pdb_path = directory / f"snapshot_{snapshot_index:03d}.pdb"
            writer = PDBIO()
            writer.set_structure(deformed)
            writer.save(str(pdb_path), _DomainSelect(ranges))
            pdb_paths.append(pdb_path)

        def detect(pdb_path):
            return (
                detector(pdb_path, tuple(sorted(ranges)))
                if detector is not None
                else _run_fpocket_pdb(pdb_path, tuple(sorted(ranges)), command)
            )
        if workers == 1:
            snapshots = tuple(detect(path) for path in pdb_paths)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                snapshots = tuple(executor.map(detect, pdb_paths))
    return nodes, tuple(snapshots)

def find_stable_ensemble_pockets(
    mmcif: Path,
    chain,
    start=None,
    end=None,
    *,
    mode_count: int = 32,
    amplitudes=(2.0, 2.5),
    anm_cutoff: float = 12.0,
    overlap_threshold: float = 0.5,
    executable: str = "fpocket",
    detector=None,
    workers: int = 4,
) -> tuple[Pocket, ...]:
    """Find cavity families reproduced across two amplitudes of each ENM mode."""

    amplitude_tuple = tuple(float(value) for value in amplitudes)
    if len(amplitude_tuple) != 2:
        raise ValueError("stable cavity discovery requires exactly two amplitudes")
    _, snapshots = _sample_enm_pockets(
        mmcif,
        chain,
        start,
        end,
        mode_count=mode_count,
        amplitudes=amplitude_tuple,
        anm_cutoff=anm_cutoff,
        executable=executable,
        detector=detector,
        workers=workers,
    )
    event_count = 2 * mode_count

    def plausible(snapshot):
        return tuple(
            pocket
            for pocket in snapshot
            if 4 <= len(pocket.residue_numbers) <= 60
            and 30.0 <= pocket.volume <= 2500.0
        )

    scouts = tuple(plausible(snapshot) for snapshot in snapshots[1 : 1 + event_count])
    refinements = tuple(plausible(snapshot) for snapshot in snapshots[1 + event_count :])
    merged = merge_stable_pocket_families(
        scouts,
        refinements,
        stability_threshold=overlap_threshold,
        family_overlap=overlap_threshold,
    )
    if not merged:
        raise ValueError("no cavity reproduced across the two ENM amplitudes")
    return merged

def _hop_rows(connectivity: np.ndarray, adjacency: np.ndarray, sources) -> np.ndarray:
    """Return within-hop percentile rows for a small set of source residues."""

    hops = shortest_path(adjacency, directed=False, unweighted=True)
    rows = []
    for source in sources:
        values = np.zeros(len(adjacency), dtype=float)
        for hop in np.unique(hops[source]).astype(int):
            if hop == 0:
                continue
            shell = np.flatnonzero(hops[source] == hop)
            values[shell] = rankdata(connectivity[source, shell]) / len(shell)
        rows.append(values)
    return np.asarray(rows)

def rank_stable_pockets(
    residues,
    adjacency: np.ndarray,
    pockets,
    *,
    source_indices,
    orthosteric_indices,
    excluded_indices,
    raw_scores: np.ndarray,
    hop_scores: np.ndarray,
    hop_explained_variance: float,
    source_raw_scores: np.ndarray | None = None,
    source_hop_scores: np.ndarray | None = None,
    hop_threshold: float = 0.8,
    top_k: int = 5,
) -> PocketRanking:
    """Rank amplitude-stable cavity families with one target-independent rule.

    Pocket selection averages within-hop transport across all orthosteric launch
    residues. ENM persistence only resolves transport near-ties. Residue ranking
    uses the apo-derived raw/hop reliability rule. No cavity volume, protein name,
    or validation label enters either decision.
    """

    residue_tuple = tuple(residues)
    candidate_pockets = tuple(pockets)
    graph = np.asarray(adjacency, dtype=float)
    raw = np.asarray(raw_scores, dtype=float)
    hop = np.asarray(hop_scores, dtype=float)
    sources = tuple(dict.fromkeys(int(index) for index in source_indices))
    orthosteric = set(int(index) for index in orthosteric_indices)
    excluded = set(int(index) for index in excluded_indices)
    node_count = len(residue_tuple)
    if (
        graph.shape != (node_count, node_count)
        or raw.shape != (node_count,)
        or hop.shape != (node_count,)
        or not np.isfinite(raw).all()
        or not np.isfinite(hop).all()
        or not sources
        or top_k < 1
    ):
        raise ValueError("stable pocket ranking inputs must match the residue graph")

    lookup = {residue.residue_number: index for index, residue in enumerate(residue_tuple)}
    orthosteric_shell = set(orthosteric)
    if orthosteric:
        orthosteric_shell.update(
            np.flatnonzero(np.any(graph[list(orthosteric)] > 0.0, axis=0))
        )
    use_region_consensus = (
        source_raw_scores is not None and source_hop_scores is not None
    )
    candidates = []
    for pocket in candidate_pockets:
        family_nodes = tuple(
            lookup[number] for number in pocket.residue_numbers if number in lookup
        )
        valid_observations = []
        for observation in pocket.observations or (pocket.residue_numbers,):
            observation_nodes = tuple(
                lookup[number] for number in observation if number in lookup
            )
            if (
                len(observation_nodes) >= top_k
                and not orthosteric.intersection(observation_nodes)
                and len(set(observation_nodes) & orthosteric_shell)
                / len(observation_nodes)
                < 0.5
            ):
                valid_observations.append(observation_nodes)
        if not valid_observations:
            continue
        nodes = tuple(sorted(set().union(*(set(group) for group in valid_observations))))
        eligible = tuple(node for node in nodes if node not in excluded)
        family_overlap = (
            len(set(family_nodes) & orthosteric_shell) / len(family_nodes)
            if family_nodes
            else 1.0
        )
        family_eligible = (
            tuple(
                node
                for node in family_nodes
                if node not in excluded and node not in orthosteric
            )
            if family_overlap < 0.5
            else eligible
        )
        if len(eligible) < top_k:
            continue

        def strongest_mean(values, group):
            count = min(top_k, len(group))
            return float(np.sort(values[list(group)])[-count:].mean())

        transport_p = 1.0
        if not use_region_consensus:
            observation_p_values = []
            for observation_nodes in valid_observations:
                nulls = matched_spatial_null_pockets(
                    residue_tuple,
                    pocket_indices=observation_nodes,
                    source_indices=sources,
                )
                transport = strongest_mean(raw, observation_nodes)
                null_transport = [strongest_mean(raw, group) for group in nulls]
                observation_p_values.append(
                    (1 + sum(value >= transport for value in null_transport))
                    / (len(null_transport) + 1)
                )
            transport_p = min(observation_p_values)
        candidates.append(
            {
                "pocket": pocket,
                "nodes": nodes,
                "eligible": eligible,
                "family_nodes": family_nodes,
                "family_eligible": family_eligible,
                "transport_p": transport_p,
            }
        )
    if not candidates:
        raise ValueError("no stable distal pocket contains enough eligible residues")

    for candidate in candidates:
        recurrence_p = (
            1
            + sum(
                other["pocket"].persistence >= candidate["pocket"].persistence
                for other in candidates
            )
        ) / (len(candidates) + 1)
        product = candidate["transport_p"] * recurrence_p
        combined_p = min(1.0, product * (1.0 - np.log(product)))
        candidate["ranking_p"] = (
            combined_p
            if hop_explained_variance >= hop_threshold
            else candidate["transport_p"]
        )
    if use_region_consensus:
        candidate_groups = tuple(candidate["eligible"] for candidate in candidates)
        region_scores = orthosteric_region_pocket_scores(
            source_raw_scores,
            source_hop_scores,
            candidate_groups,
            hop_dominated=True,
            top_k=top_k,
        )
        for candidate, region_score in zip(candidates, region_scores):
            candidate["region_score"] = float(region_score)
            # Persistence may resolve a transport near-tie, but its maximum
            # contribution is one candidate-rank interval and cannot dominate QW.
            candidate["selection_score"] = float(region_score) + (
                candidate["pocket"].persistence / (len(candidates) + 1.0)
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate["selection_score"],
                -candidate["pocket"].persistence,
                candidate["pocket"].number,
            )
        )
        ranking_method = "orthosteric_region_hop_ctqw"
    else:
        candidates.sort(
            key=lambda candidate: (
                candidate["ranking_p"],
                -candidate["pocket"].persistence,
                candidate["pocket"].number,
            )
        )
        ranking_method = "matched_null_ctqw"
    selected_candidate = candidates[0]
    selected_pocket = selected_candidate["eligible"]
    if (
        use_region_consensus
        and len(candidates) > 1
        and all(
            candidate["pocket"].observations
            or candidate["pocket"].center is not None
            for candidate in candidates
        )
    ):
        centers = np.asarray(
            [
                np.stack(
                    [residue_tuple[index].coordinate for index in candidate["family_nodes"]]
                ).mean(axis=0)
                if candidate["pocket"].observations
                else candidate["pocket"].center
                for candidate in candidates
            ],
            dtype=float,
        )
        candidate_groups = tuple(candidate["family_eligible"] for candidate in candidates)
        field_scores = orthosteric_region_pocket_scores(
            np.asarray(source_raw_scores, dtype=float),
            np.asarray(source_hop_scores, dtype=float),
            candidate_groups,
            hop_dominated=True,
            top_k=top_k,
        )
        direct_pocket, _ = select_mechanically_supported_cavity_field(
            centers,
            candidate_groups,
            transport_scores=field_scores,
            mechanical_scores=np.zeros(len(candidates)),
            direct_index=0,
            direct_pocket=candidates[0]["eligible"],
        )
        direct_is_singleton = int(np.count_nonzero(
            np.linalg.norm(centers - centers[0], axis=1) <= 8.0
        )) == 1
        used_recurrence = False
        selected_pocket = direct_pocket
        if direct_is_singleton:
            anm_response, _, _ = anm_fluctuation_connectivity(
                residue_tuple, cutoff=10.0, mode_count=min(100, node_count)
            )
            mechanical_rows = _hop_rows(
                anm_response, graph, tuple(sorted(orthosteric))
            )
            mechanical_scores = np.asarray(
                [
                    np.mean(
                        [
                            strongest_mean(row, candidate["eligible"])
                            for row in mechanical_rows
                        ]
                    )
                    for candidate in candidates
                ]
            )
            selected_pocket, used_recurrence = select_mechanically_supported_cavity_field(
                centers,
                candidate_groups,
                transport_scores=field_scores,
                mechanical_scores=mechanical_scores,
                direct_index=0,
                direct_pocket=candidates[0]["eligible"],
            )
        if used_recurrence:
            ranking_method += "_mechanical_recurrence"

    return PocketRanking(
        selected_pocket=selected_pocket,
        candidate_pockets=tuple(c["eligible"] for c in candidates),
        candidate_geometry_scores=tuple(c["pocket"].geometry_score for c in candidates),
        candidate_quantum_scores=tuple(c.get("selection_score", -np.log(c["ranking_p"])) for c in candidates),
        ranking_method=ranking_method,
    )
