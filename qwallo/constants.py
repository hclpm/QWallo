"""Every fixed parameter of the pipeline, declared in one place.

Values are the ones the reference implementation used. Changing any of them
changes the result, so each stage records the value it ran with.
"""

import numpy as np

# --- network ---------------------------------------------------------------
CONTACT_CUTOFF = 7.8          # Angstrom, C-alpha pair distance for a graph edge

# --- CTQW ------------------------------------------------------------------
DEFAULT_TIMES = np.geomspace(0.02, 20.0, 120)    # main connectivity time grid
POCKET_TIMES = np.linspace(0.02, 200.0, 401)     # cavity-ranking time grid
CTQW_DECAY_TIME = 14.0                           # exp(-t/tau) weight on POCKET_TIMES

# --- source (seed) selection ----------------------------------------------
SOURCE_SPAN = 10              # sequence window masked around a candidate source
CONCENTRATION_SIZE = 5        # distal residues summed for distal concentration
SOURCE_COUNT = 1              # launch residues kept

# --- cavity ensemble -------------------------------------------------------
POCKET_MODES = 32             # low ANM modes sampled
POCKET_AMPLITUDES = (2.0, 2.5)
ANM_CUTOFF = 12.0             # Angstrom, ANM contact cutoff for mode generation
OVERLAP_THRESHOLD = 0.5       # Jaccard for amplitude reproduction and family merge

# --- residue optimisation --------------------------------------------------
TOP_K = 5                     # residues in the submitted hit list
PAIR_WEIGHT = 0.1             # positive penalises redundant pairs

# --- site definition -------------------------------------------------------
CONTACT_RADIUS = 4.5          # Angstrom, heavy-atom contact to the binding partner
