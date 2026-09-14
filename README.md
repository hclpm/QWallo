# QWallo

### Allosteric Site Prediction by Quantum Walk Propagation on Residue Networks

CTQW propagation + elastic-network cavity ensemble + exact subset optimisation, on seven targets.

Predict distal, potentially allosteric residues from an **apo** protein structure
and a supplied orthosteric annotation, using a continuous-time quantum walk on
the residue contact network, an elastic-network cavity ensemble, and an exact
fixed-cardinality subset optimisation.

The repository is organised as **numbered stages**. Each stage reads only the
outputs of earlier stages and writes only its own directory, so any stage can be
re-run alone and the dependency order is visible in the filesystem.

```
data/<target>/<apo>.cif                 validation_data/<target>/<holo>.cif
        |                                          |
        v                                          v
00_structures   parse C-alpha nodes, fingerprint inputs
01_sites        active site (seed)  <-  annotation prior
                allosteric site (ground truth)  <-  holo binding-partner contact
02_network      C-alpha 7.8 A binary adjacency + Laplacian
03_propagation  CTQW, hop normalisation            -> connectivity.csv  (submission 1)
04_pockets      ENM cavity ensemble -> cavity selection -> lining QUBO
                                                     -> hit_list.csv    (submission 2)
05_evaluation   tier 1 random background -> tier 2 matched fake pockets -> tier 3 hits@5
06_baselines    RWR restart sweep, coined quantum walk
07_robustness   contact-cutoff sweep, time convention, pair weight
08_report       one summary table + reproduction check
```

Computation is entirely classical. The quantum walk is evaluated by exact
eigendecomposition; no Qiskit installation and no QPU are involved, and nothing
here demonstrates a quantum speed-up. The included numbers are retrospective
development results on seven targets, not evidence of generalisation.

---

## Install

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

`fpocket` is required only by the full cavity-discovery path in stage 04. Install
it from your package manager and put it on `PATH`, or pass `--fpocket /path/to/fpocket`.
The replay path needs no external tool.

## Run

```bash
# full prediction, rebuilding the cavity ensemble from the apo structure
./run_all.sh all

# reuse the published cavity; no fpocket needed
./run_all.sh all --replay
```

Any stage can be run alone, for one target or for all:

```bash
python stages/03_propagation.py kras --time-convention converged
python stages/07_robustness.py all
python stages/08_report.py --work work --reference reference
```

Outputs land under `work/<target>/<stage>/`. `work/` is git-ignored.

---

## Method, in the order the stages run

**Nodes and edges.** One node per residue at its C-alpha. An edge exists when
two C-alpha atoms lie within 7.8 A. Edges are undirected and unweighted, and no
sequence-adjacency filter is applied: consecutive residues fall inside the cutoff
automatically, and removing them would fragment the graph. Hetero atoms are
discarded at parse time, so no ligand, ion, solvent, or cofactor can leak from
the input structure into a prediction.

**Sites.** Both sites use one criterion - a residue belongs to a site when it has
a heavy atom within 4.5 A of any heavy atom of that site's **binding partner**.
"Binding partner" rather than "ligand" because in the PDB data model a ligand is
a non-polymer chemical component, which excludes the DNA, RNA, and peptide chains
that define sites on many targets. The active site is supplied as an annotation
prior in `config/targets.py`; the allosteric site is derived from the holo
structure and is never read by a prediction stage.

**Propagation.** `H = L / lambda_max`, so time is dimensionless. Two matrices are
produced from the same graph because the consumers need different windows: the
submitted connectivity on a 120-point geometric grid over 0.02-20, and a
cavity-ranking transport matrix on a 401-point linear grid over 0.02-200 weighted
by `exp(-t/14)`. `--time-convention` selects the read-out: the time average, the
instantaneous distribution at one large time, or the closed-form infinite-time
average. A unitary walk never decoheres, so these are three readings of one
operator, not successive approximations.

**Hop normalisation.** Raw occupancy falls off with graph distance from the seed,
and an allosteric site is not distinguished by distance. Residues are therefore
ranked only against others at the same hop count. The reported
`hop_explained_variance` is the share of log-connectivity variance explained by
hop-shell medians - how distance-dominated the raw score was.

**Cavity ensemble.** 32 low ANM modes are displaced in both directions at two RMS
amplitudes, giving 129 snapshots including the reference; fpocket runs on each. A
cavity seen at the first amplitude must reappear at the second to count as
stable, and stable observations are merged into families by lining overlap. This
is what allows a cavity that is closed in the deposited structure to be proposed
at all.

**Cavity selection.** Families are scored by within-hop CTQW transport averaged
over the whole orthosteric region. Persistence enters only as a tie-break bounded
to one candidate-rank interval. fpocket's own score and the cavity volume do not
participate.

**Residue subset.** Inside the selected cavity the objective rewards cavity-wall
support (the alpha-sphere percentile) and penalises redundant pairs (CTQW pair
rank times a Gaussian distance taper, weight 0.1). All `C(m, 5)` subsets are
enumerated exactly; runtime is `O(binomial(m,5))`, not a speed-up.

---

## Results

Seven targets, replay path, default parameters.

| target | nodes | edges | seeds | positives | cavity | tier 1 lift | tier 2 lift | hits@5 |
|---|---|---|---|---|---|---|---|---|
| KRAS G12C | 169 | 779 | 8 | 17 | 22 | 5.4 | not evaluable | **5/5** |
| BCR-ABL1 | 290 | 1266 | 21 | 20 | 46 | 12.0 | 2.27 | **5/5** |
| Cardiac myosin | 775 | 3271 | 1 | 12 | 35 | 42.6 | 3.53 (flagged) | **5/5** |
| MAPK14 | 345 | 1507 | 11 | 16 | 26 | 10.5 | 2.32 | 3/5 |
| PTPN1 | 283 | 1323 | 8 | 9 | 23 | 10.8 | 3.12 (flagged) | 2/5 |
| PDK2 | 357 | 1625 | 15 | 12 | 20 | 22.2 | 5.71 (flagged) | 4/5 |
| PTPN11 | 499 | 2339 | 9 | 17 | 63 | 17.5 | 5.68 (flagged) | 4/5 |

**28/35** summed. `lift` is AUPRC divided by its floor; 1.0 is chance.

### Reproduction

`08_report` writes `reproduction.csv`, comparing the rebuilt submission against
the published one element by element.

| check | result |
|---|---|
| top-5 residues, 7/7 targets | identical |
| hits@5, 7/7 targets | identical |
| N x N connectivity matrix (KRAS) | max absolute difference 2.2e-16 |
| KRAS from **full** fpocket discovery | identical cavity (22 residues), identical ranking rule, identical top-5 |

The full-discovery run found 27 cavity families where the published record has
28; the extra family is not the selected one, so the answer is unchanged. Full
discovery was verified on KRAS only.

### Baselines - the honest read

Stage 06 compares operators at the **residue** level over the whole non-seed
candidate set, using the hop-normalised propagation score only. The selected
cavity is deliberately not used, because it was chosen with CTQW and feeding a
baseline through that choice would not be a control.

| target | CTQW | best RWR (restart) | coined QW |
|---|---|---|---|
| KRAS | 1.26 | 1.29 (0.50) | **2.48** |
| BCR-ABL1 | 1.91 | 1.79 (0.10) | **2.37** |
| myosin | **1.48** | 1.20 (0.05) | 1.34 |
| MAPK14 | 0.67 | 0.68 (0.50) | **2.26** |
| PTPN1 | 0.66 | 0.66 (0.05) | 0.97 |
| PDK2 | 1.24 | **1.65** (0.50) | 1.06 |
| PTPN11 | 0.98 | 0.93 (0.50) | 1.00 |

Two things follow, and neither is favourable to the continuous-time walk.
**CTQW and a classical random walk with restart are within 0.1 of each other on
six of seven targets**, so at this level the quantum walk carries no information
the classical diffusion lacks. And the **coined** walk - a different quantum
operator on the same graph - is the best of the three on three targets, so
"quantum walk" is not the active ingredient; the specific operator matters more.

### Robustness

| axis | swept | effect on the submitted five |
|---|---|---|
| contact cutoff | 4.5 - 8.5 A, 9 values | **no change on any target** |
| time convention | average / converged / infinite limit | **no change on any target** |
| pair weight | 0.0, 0.05, 0.1, 0.2 | no change on 5 of 7; PTPN11 changes at 0.2 |
| pair weight | 0.5 | KRAS, BCR-ABL1 lose one hit; PTPN11 loses one |

This invariance is a finding, not a reassurance. With the cavity fixed, the five
residues are decided by cavity-wall geometry: the CTQW pair term at its reference
weight of 0.1 changes nothing, and setting it to zero changes nothing either. The
walk's real influence is on **cavity selection**, which the replay path bypasses -
so a cutoff sweep through full discovery, where the cavity ranking can move, is
the experiment this table does not yet contain.

### Evaluation tiers and their self-check

Tier 1 is weak evidence by construction: an allosteric site is buried and far
from the active site by definition, so a score measuring only those two
properties already beats a random background. Tier 2 therefore ranks the
positives against pooled residues of control pockets whose centres match the true
site in burial **and** in hop distance from the seed - two separate hard gates,
never summed.

Four control scores run through the identical path, and they are reported before
the result:

| control | expected | meaning if it fails |
|---|---|---|
| `perfect` | far above 1 | the ceiling is reachable |
| `random` (mean of 200 draws) | near 1 | no built-in bias |
| `burial` | near 1 | the burial gate worked |
| `neg_seed_dist` | near 1 | the distance gate worked |

`burial` or `neg_seed_dist` above 1.5 means the gate admitted a skewed control
set and the tier-2 number is **not interpretable**. That happens on four of seven
targets here, and stage 05 records it as a warning rather than reporting the
number silently. KRAS yields only 10 matched control residues even at the widest
tolerance and is reported **not evaluable** instead of being given a value.
`random` sits at 1.10-1.35 with a standard deviation of 0.17-0.45: average
precision is upward-biased when the matched scope holds only a few dozen
residues, which is a limit of the tier-2 scope size, not of the score.

---

## Output files

| file | stage | contents |
|---|---|---|
| `nodes.csv` | 00 | matrix index, residue id, coordinates |
| `structures.json` | 00 | input SHA-256 for apo and holo |
| `<target>_nodes.csv` | 01 | `site_class` in {orthosteric, allosteric, shared, unlabelled} |
| `sites.json` | 01 | both site definitions and their sources |
| `adjacency.npy`, `network.json` | 02 | graph and its statistics |
| `connectivity.csv` | 03 | **submission 1** - N x N CTQW matrix, zero diagonal |
| `scores.csv` | 03 | raw and hop-normalised per-residue score |
| `hit_list.csv` | 04 | **submission 2** - the five residues |
| `residue_scores.csv` | 04 | per-residue optimisation score and cavity eligibility |
| `residue_qubo.csv`, `..._variables.csv` | 04 | upper-triangular objective and its index map |
| `pocket_families.json`, `method.json` | 04 | every detected family, parameters, energy, provenance |
| `evaluation.json`, `ranking.csv`, `fake_pockets.csv` | 05 | three tiers, controls, warnings |
| `baselines.csv/json` | 06 | every operator and restart probability |
| `cutoff_sweep.csv`, `time_convention.csv`, `pair_weight.csv` | 07 | robustness sweeps |
| `summary.csv`, `reproduction.csv`, `report.md` | 08 | aggregate |

`residue_scores.csv` carries an **optimisation score**, not an occupancy: `-1`
outside the cavity, a normalised inclusion utility inside it, `+1` added for the
selected five. Tier-1 lift is therefore dominated by cavity membership; stage 06
is where the propagation score is judged on its own.

---

## Limitations and what is not here

- **Cavity selection is the bottleneck, not residue ranking.** On the targets that
  miss, the answer residues are in the protein's cavity set but not in the
  selected cavity.
- **Full discovery is verified on one target.** The other six replay a published
  cavity. Re-running full discovery for all seven is the obvious next step.
- **Three published cavities came from a code path that no longer exists.** The
  upstream records for MAPK14, PTPN1, and PDK2 name a `cohesive_subset` step that
  is absent from this source tree; their replays reproduce, a full re-run may not.
- **Tier 2 is not interpretable on four of seven targets** and not evaluable on
  one, for the reasons the warnings state.
- **No hardware stage.** Circuit compilation, qubit-count reduction, and noise
  analysis are not implemented here.
- **Ground truth depends on the holo structure chosen.** `config/targets.py`
  records the exact structure and binding partner per target; changing either
  changes the labels and makes numbers incomparable.

## Layout

```
config/targets.py      seven target presets: apo, domain, orthosteric prior, holo, partner
qwallo/                 library
  constants.py         every fixed parameter in one place
  structure.py         mmCIF -> C-alpha nodes, contact network
  propagation.py       time-averaged CTQW
  pipeline.py          residue ids, source selection, hop normalisation
  anm.py               ANM Hessian and modes
  pockets.py           cavity ensemble, family merging, cavity ranking
  subset.py            lining QUBO and exact enumeration
  sites.py             binding-partner site definition
  baselines.py         RWR, coined quantum walk
  evaluation.py        three-tier metrics, matched fake pockets
  workspace.py         stage paths and IO
stages/00_..08_        one script per stage
reference/<target>/    published method.json, hit_list.csv, evaluation.json
data/, validation_data/  apo and holo mmCIF inputs
```
