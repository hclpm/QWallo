# Implementation Reference

This document describes the current QAI MVP implementation: how an apo protein
structure is converted into a residue-level CTQW connectivity map, how distal cavity
families are generated and selected, and how the final five residues are chosen.

The implementation is a classical reference pipeline. Quantum algorithms such as
SKQD and QAOA are planned extensions and are not executed by the current MVP.

## 1. Runtime modules

```text
run.py
  Benchmark presets and common prediction entry point.

predict.py
  End-to-end apo prediction orchestration and output generation.

validate.py
  Independent holo ligand-contact evaluation.

connectivity/structure.py
  mmCIF parsing and C-alpha contact-network construction.

connectivity/maps.py
  Exact time-averaged CTQW connectivity.

connectivity/pipeline.py
  Orthosteric-source handling and hop-normalized CTQW scores.

connectivity/anm.py
  Directional ANM Hessian and mechanical-response calculations.

connectivity/pockets.py
  ANM conformer generation, fpocket integration, cavity-family construction,
  and distal pocket selection.

connectivity/subset.py
  Pocket-lining unary scores, CTQW pair terms, and exact fixed-cardinality QUBO.
```

`run.py` maps a target name to its apo structure, selected domain, and supplied
orthosteric residues. Every target is then passed to the same `predict.main()`
implementation. Protein names and holo validation labels are not used by the
prediction scoring functions.

Prediction and validation are separate execution paths:

```text
                    PREDICTION

apo structure + orthosteric annotation
            |
            v
      C-alpha graph
            |
            v
     CTQW connectivity
            |
            v
    ANM conformer ensemble
            |
            v
   fpocket cavity families
            |
            v
     distal pocket ranking
            |
            v
   pocket-local residue QUBO
            |
            v
        Top-5 residues


                    VALIDATION

saved prediction + holo structure
            |
            v
     ligand-contact labels
            |
            v
    Hits@5 / AUPRC / AUROC
```

`predict.py` can therefore run without `validation_data/`.

---

## 2. Residue identity and structural representation

`structure.parse_mmcif_residues()` reads the first model of the selected mmCIF
structure and creates one `ResidueNode` per standard amino acid with a resolved
C-alpha atom.

```python
@dataclass(frozen=True)
class ResidueNode:
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    coordinate: np.ndarray
```

Ligands, solvent, ions, cofactors, and nonstandard residues are removed before graph
construction.

The position of a `ResidueNode` in the returned list is the internal matrix index.
It is not the PDB author residue number. External residue IDs use

```text
CHAIN:AUTHOR_NUMBER[INSERTION_CODE]
```

for example `A:60`.

The cavity implementation accepts one chain, or several chains whose author
residue numbers do not collide, and rejects insertion-code domains, because
fpocket family membership is stored using author residue numbers. A multimer
target is given one window per chain, as `--chain A --range A:897-984 --chain B
--range B:202-284`.

---

## 3. C-alpha network and CTQW connectivity

`structure.build_uniform_contact_network()` constructs the binary contact graph

```text
A_ij = 1  if 0 < ||r_i - r_j|| <= 7.8 Å
A_ij = 0  otherwise
```

and requires the resulting graph to be connected.

The CTQW reference kernel is implemented in
`maps.time_averaged_ctqw_connectivity()`.

For adjacency matrix `A`,

```text
L = D - A
H = L / lambda_max(L)

U(t) = exp(-i H t)
C_ij = sum_t w_t |U_ij(t)|^2 / sum_t w_t
```

The full exported connectivity matrix uses

```python
DEFAULT_TIMES = np.geomspace(0.02, 20.0, 120)
```

with equal weights.

Implementation details:

- `numpy.linalg.eigh()` diagonalizes the real symmetric Laplacian;
- `lambda_max` normalization makes the time axis dimensionless;
- no molecular timescale is implied;
- the self-return diagonal is removed after averaging;
- the final matrix is symmetrized and clipped to `[0, 1]`.

Because the diagonal is removed, `connectivity.csv` is a connectivity descriptor
rather than a row-stochastic probability matrix.

---

## 4. Orthosteric conditioning

The supplied orthosteric annotation is resolved by `resolve_sources()`.

`select_distal_sources()` evaluates each supplied orthosteric residue using its
concentrated distal CTQW response. For source `s`, residues in the orthosteric set
and its local sequence window are removed before the five strongest distal
connectivities are summed.

```text
source score
    = sum of the five strongest distal CTQW values
```

The strongest candidate becomes the representative launch source.

This representative source is useful for source-level diagnostics, but final pocket
selection uses all supplied orthosteric residues.

### Hop normalization

Raw connectivity can be dominated by graph distance. Therefore
`hop_normalized_source_scores()` computes shortest-path distance on the contact graph
and ranks residues only against residues in the same hop shell.

```text
source
  |
  +-- 1-hop residues -> percentile ranks
  |
  +-- 2-hop residues -> percentile ranks
  |
  +-- 3-hop residues -> percentile ranks
  ...
```

The resulting scores measure whether CTQW transport is unusually strong relative to
other residues at comparable network distance.

---

## 5. ANM conformer ensemble and cavity families

A single apo structure can fragment or hide a shallow cavity. The implementation
therefore generates a controlled ensemble of nearby structures before pocket ranking.

`anm.build_anm_hessian()` constructs the standard directional `3N x 3N` ANM Hessian.
`_low_anm_modes()` obtains low-frequency non-rigid modes using dense eigendecomposition
for small systems and sparse `eigsh()` for larger systems.

`find_stable_ensemble_pockets()` uses:

```text
ANM cutoff       12 Å
retained modes   32
directions       + / -
RMS amplitudes   2.0 Å and 2.5 Å
```

This gives

```text
1 native structure
+ 32 modes × 2 directions × 2 amplitudes
= 129 snapshots
```

The C-alpha displacement of each residue is applied to all atoms belonging to that
residue, and fpocket is run on each temporary structure.

fpocket observations are filtered to retain plausible cavities:

```text
lining residues  4–60
volume           30–2500 Å³
```

Observations generated at the two amplitudes are matched using lining-residue
Jaccard overlap. The default threshold is

```text
Jaccard >= 0.5
```

Repeated observations are merged into `Pocket` families containing:

```python
Pocket(
    residue_numbers,
    center,
    persistence,
    residue_persistence,
    void_support,
    observations,
    volume,
    geometry_score,
)
```

`void_support` is the residue-level cavity-wall evidence later used by the final
subset objective.

---

## 6. Distal pocket selection

`rank_stable_pockets()` converts the cavity ensemble into one final distal pocket.

Candidate observations are first removed when they

- contain an orthosteric residue;
- are dominated by the orthosteric one-hop shell; or
- do not retain at least `top_k` eligible residues after the local source mask.

Pocket ranking uses a second CTQW integration:

```python
POCKET_TIMES = np.linspace(0.02, 200.0, 401)
CTQW_DECAY_TIME = 14.0
```

with

```text
w(t) = exp(-t / 14)
```

For every supplied orthosteric residue, the implementation computes hop-normalized
transport and summarizes a candidate cavity by its strongest `top_k` residue values.

The final CTQW pocket signal is therefore based on the complete orthosteric region:

```text
orthosteric residue 1 ─┐
orthosteric residue 2 ─┼─> within-hop CTQW -> candidate-pocket score
orthosteric residue 3 ─┘
                                  |
                                  v
                           average over sources
```

Family persistence is included only as a bounded secondary adjustment. fpocket's
native geometry score and cavity volume are stored as metadata but do not define the
functional ranking.

### Mechanical cavity refinement

Closely positioned cavity families may represent fragments of one larger binding
region.

When the direct CTQW winner is isolated, the implementation can calculate a
directional ANM perturbation-response matrix using up to 100 modes and compare the
mechanical support of nearby recurrent cavity fields.

This step can modify the selected **pocket**.

It does not create a target-specific residue-ranking path; after pocket selection,
all targets use the same `subset.py` objective.

---

## 7. Pocket-local residue optimization

The final residue-selection layer is implemented in `subset.py`.

### 7.1 Lining unary

The selected pocket is matched to the detected cavity family with the largest
Jaccard overlap.

For each eligible residue `i`, its cavity-wall support `v_i` is converted to a
tie-aware percentile:

```text
g_i = rank(v_i within the selected pocket) / m
```

where `m` is the number of eligible pocket residues.

`g_i` is the unary utility used by the final optimization.

The default implementation uses cavity-wall support directly. No protein-specific
CTQW or ANM residue objective is selected at this stage.

### 7.2 Pair term

The pair term combines three quantities:

1. the two residues must both have useful lining support;
2. their CTQW pair connectivity should be large;
3. their C-alpha coordinates should be spatially close.

For pair `(i, j)`,

```text
P_ij =
    min(g_i, g_j)
    * c_ij
    * exp[-0.5 * (d_ij / 7.8)^2]
    / (k - 1)
```

where

```text
c_ij = percentile rank of C_ij among candidate pairs
d_ij = C-alpha distance
```

The final objective is

```text
E(x)
    = -sum_i g_i x_i
      + beta sum_{i<j} P_ij x_i x_j

subject to

sum_i x_i = k
```

with

```text
k = 5
beta = 0.1
```

by default.

With the current sign convention, positive `beta` penalizes selecting strongly
CTQW-coupled, spatially nearby residues together. The pair term therefore acts as
a bounded redundancy penalty around the stronger cavity-lining unary signal.

### 7.3 Matrix convention and solver

`residue_qubo.csv` is upper triangular:

```text
Q_ii = -g_i
Q_ij = beta P_ij,   i < j
Q_ij = 0,           i > j
```

and energy is evaluated as

```text
E = x.T @ Q @ x
```

The cardinality constraint is not encoded inside the matrix.

`solve_pocket_qubo()` enforces `sum(x)=k` directly and enumerates all `k`-subsets in
batches of 8192.

For `m` candidate residues, complexity is approximately

```text
O( C(m,k) k² )
```

The current solver provides an exact classical optimum for the pocket-local problem.
It is intended as the reference solution for a future constraint-preserving QAOA
experiment.

---

## 8. Prediction outputs and provenance

A full prediction writes:

```text
connectivity.csv
    N x N time-averaged CTQW connectivity matrix

residue_qubo.csv
    upper-triangular pocket-local QUBO

residue_qubo_variables.csv
    QUBO variable -> protein residue mapping and lining percentile

residue_scores.csv
    complete protein ranking and pocket eligibility

hit_list.csv
    final five-residue subset

method.json
    execution parameters, selected pocket, cavity evidence,
    QUBO energy, input hash, and execution scope
```

The compatibility column `quantum_score` in `residue_scores.csv` is not a raw CTQW
probability or QPU measurement.

For pocket residue `i`, the solver records the best achievable objective among
feasible subsets containing that residue. These inclusion utilities are normalized,
and residues belonging to the returned optimal subset receive an additional `+1`
priority so that the certified optimum appears first in the exported ranking.

---

## 9. Replay mode

Full cavity discovery is substantially more expensive than the final subset
optimization because it requires ANM conformer generation and repeated fpocket runs.

`predict.py --replay` therefore supports a fixed-pocket execution mode.

Replay loads:

```text
connectivity.csv
selected pocket
detected cavity families
```

from an existing run and recomputes only

```text
lining unary
    ->
pair terms
    ->
QUBO optimum
    ->
residue scores
```

Before reuse, replay checks:

- ordered residue identities;
- orthosteric annotations;
- input structure SHA-256 when available.

`method.json` records whether the output was produced by

```text
full apo prediction
```

or

```text
fixed-pocket replay
```

so the two execution scopes are not confused.

---

## 10. Validation

`validate.py` is an independent post-prediction program.

It reads the previously written `residue_scores.csv`, opens the corresponding holo
structure, and identifies all standard residues in chain A having a heavy atom within

```text
4.5 Å
```

of any heavy atom of the specified validation ligand.

Prediction and validation residues are matched using author residue identity.

The evaluator reports:

```text
residue_count
positive_count
random_auprc
AUPRC
AUROC
Hits@5
```

AUPRC is evaluated at distinct score thresholds so tied residues are evaluated
together rather than being affected by CSV ordering.

The current benchmark assumes compatible apo/holo author numbering and does not
perform sequence alignment.

---

## 11. Execution scope

The current MVP executes:

```text
Full CTQW connectivity
    exact classical Laplacian eigendecomposition

ANM conformer generation
    classical dense/sparse eigensolver

Cavity detection
    fpocket over the apo ANM ensemble

Pocket selection
    CTQW region-consensus ranking
    + optional mechanical cavity refinement

Residue optimization
    exact fixed-cardinality QUBO enumeration

Validation
    independent holo ligand-contact evaluation
```

The current repository does not execute SKQD, QAOA, or physical-QPU circuits.

Those quantum components are intended to replace well-defined classical subroutines:

```text
classical ANM low-mode solver
        ↓
       SKQD

exact pocket QUBO solver
        ↓
constraint-preserving QAOA
```

This keeps the present implementation as a complete classical reference against
which future QPU results can be evaluated.

---

## 12. Minimal call graph

```text
run.run_target
  |
  └── predict.main
        |
        ├── parse_mmcif_residues
        ├── resolve_sources
        |
        ├── build_uniform_contact_network
        ├── time_averaged_ctqw_connectivity
        |
        ├── select_distal_sources
        ├── hop_normalized_source_scores
        |
        ├── find_stable_ensemble_pockets
        │     ├── build_anm_hessian
        │     ├── _low_anm_modes
        │     └── fpocket
        |
        ├── rank_stable_pockets
        │     └── [optional] anm_fluctuation_connectivity
        |
        ├── build_lining_qubo
        ├── solve_pocket_qubo
        ├── subset_scores
        |
        └── write prediction artifacts


validate.main
  |
  └── validate
        ├── ligand_contacts
        ├── metrics
        └── evaluation.json
```