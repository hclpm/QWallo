"""Target registry: apo input, domain, orthosteric prior, holo reference.

One dictionary is the single source of truth for every stage. ``orthosteric``
is a supplied biological prior, not something the pipeline infers. ``holo`` and
``partner`` are read only by stage 01 to build ground-truth labels and by stage
05 to score; no prediction stage ever opens them.
"""

TARGETS = {
    "kras": dict(
        name="KRAS G12C", apo="4OBE.cif", chain="A", residue_range=(1, 169),
        orthosteric=(11, 12, 13, 14, 15, 16, 17, 18),
        holo="6OIM.cif", partner="MOV",
    ),
    "bcr": dict(
        name="BCR-ABL1", apo="1OPL.cif", chain="A", residue_range=(242, 533),
        orthosteric=(267, 268, 272, 275, 288, 289, 290, 305, 309, 318, 332,
                     334, 335, 336, 337, 338, 340, 389, 399, 400, 401),
        holo="5MO4.cif", partner="AY7",
    ),
    "myosin": dict(
        name="Cardiac myosin", apo="5TBY.cif", chain="A", residue_range=(6, 780),
        orthosteric=(184,),
        holo="9YR7.cif", partner="XB2",
    ),
    "mapk14": dict(
        name="MAPK14 (p38a)", apo="1R39.cif", chain="A", residue_range=(4, 351),
        orthosteric=(*range(30, 39), 53, 168),
        holo="3NEW.cif", partner="3NE",
    ),
    "ptpn1": dict(
        name="PTPN1 (PTP1B)", apo="1A5Y.cif", chain="A", residue_range=(2, 285),
        orthosteric=(181, *range(216, 222), 262),
        holo="1T48.cif", partner="BB3",
    ),
    "pdk2": dict(
        name="PDK2", apo="2BTZ.cif", chain="A", residue_range=(6, 384),
        orthosteric=(*range(251, 259), 290, *range(325, 331)),
        holo="2BU2.cif", partner="TF1",
    ),
    "ptpn11": dict(
        name="PTPN11 (SHP2)", apo="4DGP.cif", chain="A", residue_range=(3, 528),
        orthosteric=(425, *range(459, 466), 506),
        holo="5EHP.cif", partner="5OA",
    ),
}

ORDER = ("kras", "bcr", "myosin", "mapk14", "ptpn1", "pdk2", "ptpn11")
