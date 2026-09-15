<p align="center">
  <img src="assets/qwallo_logo.png" alt="QWallo — quantum walks for allostery" width="200">
</p>

# QWallo

### Allosteric Site Prediction by Quantum Walk Propagation on Residue Networks

Predict distal, potentially allosteric residues from an apo protein structure and
a supplied orthosteric annotation. Every target uses one workflow:

```text
apo structure + orthosteric residues
  → ENM conformers and fpocket cavities
  → CTQW pocket selection
  → lining score + pair term
  → exact selection of five residues
```

Computation runs on a classical CPU. No Qiskit or QPU is required. Prediction
does not read holo structures or validation labels. Included results are
retrospective development results, not evidence of generalization or quantum advantage.

## Install

Python 3.10 or newer is required. Run commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install fpocket
```

The last command is the macOS installation of fpocket. On other systems, install
fpocket and put it on `PATH`. Only full pocket discovery needs it; replay and
validation use the included files.

## Validate included results

```bash
python validate.py all
```

| Target | Hits@5 | AUPRC |
|---|---:|---:|
| KRAS | 5/5 | 0.4872 |
| BCR-ABL1 | 5/5 | 0.8956 |
| Myosin | 5/5 | 0.6608 |
| MAPK14 | 3/5 | 0.5044 |
| PTPN1 | 2/5 | 0.3547 |
| PDK2 | 4/5 | 0.7774 |
| PTPN11 | 4/5 | 0.6056 |

All results use the same lining objective and pair weight 0.1. Validation writes
`evaluation.json` using heavy-atom contacts within 4.5 Å of the specified holo
binding partner. Average precision is calculated over all parsed apo residues.

c-Myc has no holo reference structure, so it carries no ground-truth label and is
not part of `validate.py`. Its prediction is in `results/cmyc/`.

## Run

Recompute the residue QUBO on the included fixed pockets, without fpocket:

```bash
python run.py all --replay --out-root out_replay
python validate.py all --results-root out_replay
```

Rebuild the graph and cavity ensemble from the apo structure:

```bash
python run.py all --out-root out_full
python validate.py all --results-root out_full
```

Replay writes to a directory other than `results/`, which the predictor requires
so that a replay cannot overwrite the pocket it is replaying.

Replace `all` with any target in the table. Full runs generate 129 ENM snapshots
per target by default and are substantially slower than replay. Included results
are fixed-pocket replays, except c-Myc, which comes from full discovery. Full
discovery recovers the included cavity and the same five residues on KRAS,
BCR-ABL1, myosin and PDK2. On MAPK14, PTPN1 and PTPN11 the included results carry
a ranking variant that is not part of this tree, and a full run reaches a
different selection for them; `method.json` records which ranking rule produced
each result.

For a custom domain:

```bash
python predict.py data/kras/4OBE.cif \
  --chain A --range A:1-169 \
  --orthosteric A:11 --orthosteric A:12 --orthosteric A:13 \
  --orthosteric A:14 --orthosteric A:15 --orthosteric A:16 \
  --orthosteric A:17 --orthosteric A:18 \
  --out-dir results
```

A domain spanning more than one chain repeats `--chain` and `--range`, one window
per chain — this is how c-Myc is run:

```bash
python predict.py data/cmyc/1NKP.cif \
  --chain A --range A:897-984 \
  --chain B --range B:202-284 \
  --orthosteric A:898 --orthosteric A:902 ... \
  --out-dir results/cmyc
```

The chains selected this way must not share author residue numbers, because
cavity membership is stored by number; the predictor checks this and fails rather
than resolving the collision silently. Insertion codes and isolated nodes at the
7.8 Å contact cutoff are rejected. `validate.py` evaluates the included presets;
custom holo inputs require adapting its target registry.

## Targets

| Target | Apo | Domain | Orthosteric surface |
|---|---|---|---|
| KRAS G12C | 4OBE | A:1-169 | P-loop |
| BCR-ABL1 | 1OPL | A:242-533 | ATP site |
| Cardiac myosin | 5TBY | A:6-780 | nucleotide site |
| MAPK14 | 1R39 | A:4-351 | ATP site |
| PTPN1 | 1A5Y | A:2-285 | PTP loop |
| PDK2 | 2BTZ | A:6-384 | ATP site |
| PTPN11 | 4DGP | A:3-528 | PTP loop |
| c-Myc / Max | 1NKP | A:897-984 + B:202-284 | DNA contact face of Myc |

c-Myc is a heterodimer with no catalytic site, so the surface responsible for the
function being inhibited — the DNA contact face — is used as the orthosteric
annotation. One copy of the Myc/Max dimer is taken; the DNA chains and the second
copy in the asymmetric unit are not part of the node set.

## Files and outputs

`run.py` supplies presets, `predict.py` predicts or replays, and `validate.py`
evaluates. `connectivity/` contains structure parsing, CTQW, ENM, cavity selection,
and QUBO code. Apo inputs are in `data/`, holo inputs in `validation_data/`.

Results live directly in `results/<target>/`:

| File | Contents |
|---|---|
| `connectivity.csv` | N×N CTQW matrix with zero diagonal |
| `residue_scores.csv` | Residue mapping, optimization score, eligibility |
| `hit_list.csv` | Selected residues ordered by optimization score |
| `residue_qubo.csv` | Upper-triangular quadratic objective |
| `residue_qubo_variables.csv` | QUBO-to-residue mapping and lining percentile |
| `method.json` | Input hash, parameters, pocket evidence, energy, replay provenance |
| `evaluation.json` | Separate holo-contact evaluation |
