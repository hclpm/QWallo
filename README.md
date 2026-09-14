# QWallo

### Allosteric Site Prediction by Quantum Walk Propagation on Residue Networks

Predict five distal, potentially allosteric residues from an **apo** structure and a
supplied orthosteric annotation. A continuous-time quantum walk on the residue
contact network ranks cavities found in an elastic-network conformer ensemble; the
five residues are then chosen inside the selected cavity by exact enumeration.

```
00_structures   apo / holo parsing, input fingerprints
01_sites        active site (seed, annotation prior) · allosteric site (holo binding-partner contact)
02_network      C-alpha 7.8 A binary contact graph
03_propagation  CTQW + hop normalisation            -> connectivity.csv   (submission 1)
04_pockets      ANM cavity ensemble -> CTQW cavity ranking -> subset enumeration
                                                     -> hit_list.csv      (submission 2)
05_evaluation   random background -> matched fake pockets -> hits@5
06_baselines    RWR restart sweep, coined quantum walk
07_robustness   contact cutoff, time convention, pair weight
08_report       summary + reproduction check
```

## Install

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`fpocket` is needed only for full cavity discovery in stage 04; put it on `PATH` or
pass `--fpocket /path/to/fpocket`. Replay needs no external tool.

## Run

### Everything at once

```bash
./run_all.sh all                 # full prediction (needs fpocket)
./run_all.sh all --replay        # reuse the published cavity, no fpocket
./run_all.sh kras                # one target
```

### One stage at a time

Every stage takes a target name or `all`, plus `--work DIR` (default `work/`).
Each reads only earlier stages' outputs, so a stage can be re-run alone.

| stage | command |
|---|---|
| 00 | `python stages/00_structures.py all` |
| 01 | `python stages/01_sites.py all` |
| 02 | `python stages/02_network.py all --cutoff 7.8` |
| 03 | `python stages/03_propagation.py all --time-convention average` |
| 04 | `python stages/04_pockets.py all` — add `--replay-root reference` to skip discovery |
| 05 | `python stages/05_evaluation.py all` |
| 06 | `python stages/06_baselines.py all` |
| 07 | `python stages/07_robustness.py all` |
| 08 | `python stages/08_report.py --work work --reference reference` |

```bash
python stages/02_network.py kras --cutoff 5.5
python stages/03_propagation.py kras --time-convention converged   # or: limit
python stages/04_pockets.py kras --pair-weight 0.0
python stages/05_evaluation.py kras --shared-policy positive
```

`python stages/<file> --help` lists every option. Outputs go to
`work/<target>/<stage>/`; `work/` is git-ignored.

## Results

| target | nodes | cavity | AUPRC | random floor | hits@5 |
|---|---|---|---|---|---|
| KRAS G12C | 169 | 22 | 0.4872 | 0.1243 | **5/5** |
| BCR-ABL1 | 290 | 46 | 0.8956 | 0.0690 | **5/5** |
| Cardiac myosin | 775 | 35 | 0.6608 | 0.0155 | **5/5** |
| MAPK14 | 345 | 26 | 0.5044 | 0.0464 | 3/5 |
| PTPN1 | 283 | 23 | 0.3547 | 0.0318 | 2/5 |
| PDK2 | 357 | 20 | 0.7774 | 0.0336 | 4/5 |
| PTPN11 | 499 | 63 | 0.6056 | 0.0341 | 4/5 |

**28/35** summed. Ground truth is the holo binding-partner contact set listed in
`config/targets.py`; changing the holo structure changes the labels.

## Reproduction

`stages/08_report.py --reference reference` writes `reproduction.csv`.

| check | result |
|---|---|
| replay, 7/7 targets, against the published results | identical top-5 and hits; `hit_list.csv`, `residue_scores.csv`, `connectivity.csv` identical byte-for-byte |
| full cavity discovery, 7/7 targets, against the reference implementation run on the same path | identical cavity, top-5, and ranking rule |
| `residue_qubo.csv` | differs at most 1.7e-18 (double-precision last bit); selection unaffected |

One caveat belongs to the upstream results rather than to this code. The published
records for MAPK14, PTPN1 and PDK2 name a `cohesive_subset` ranking step that does
not exist in the reference source tree, so those three came from a newer version.
Re-running full discovery without it leaves PTPN1 and PDK2 unchanged but moves
MAPK14 to a different cavity (Jaccard 0.26), where it scores 0/5 and AUPRC 0.0499
against a 0.0464 floor. The reference README states that full discovery was
verified on KRAS only, so this is outside what it claims.

## Evaluation

Three scopes, from weakest to strictest:

1. **random background** — positives against every non-seed residue. Easy to pass:
   an allosteric site is buried and distant from the active site by definition.
2. **matched fake pockets** — positives against control pockets whose centres match
   the true site in burial *and* in hop distance from the seed, as two separate
   gates. Four control scores run through the same path; if `burial` or
   `neg_seed_dist` lifts above 1.5, the gate admitted a skewed control set and
   stage 05 records the scope as not interpretable. That happens on four targets;
   KRAS yields too few matched controls and is reported as not evaluable.
3. **hits@5** — the challenge output format.

Seed residues are excluded from scoring throughout.

## Baselines and robustness

Stage 06 runs RWR across restart probabilities and a coined quantum walk on the
same graph and seeds; stage 07 sweeps the contact cutoff, the time convention and
the pair weight. Both measure the **residue-level** score with the cavity held
fixed, so neither tests the CTQW cavity-ranking step that selects the cavity in the
first place. Numbers are in `06_baselines/baselines.csv` and
`07_robustness/*.csv`; an operator swap inside cavity ranking is not implemented.

## Limitations

- Cavity selection, not residue ranking, is where the misses occur: on the failing
  targets the answer residues lie in the protein's cavity set but not in the
  selected cavity.
- The `cohesive_subset` step of the published MAPK14, PTPN1 and PDK2 records is not
  available here.
- Tier 2 is not interpretable on four of seven targets and not evaluable on one.
- No hardware stage: circuit compilation, qubit-count reduction and noise analysis
  are not implemented.

## Layout

```
config/targets.py   seven target presets: apo, domain, orthosteric prior, holo, partner
qwallo/             library — structure, propagation, pipeline, anm, pockets, subset,
                    sites, baselines, evaluation, constants, workspace
stages/00_..08_     one script per stage
reference/          published method.json, hit_list.csv, evaluation.json per target
data/               apo mmCIF     validation_data/  holo mmCIF
```
