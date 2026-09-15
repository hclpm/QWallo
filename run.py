"""Run one benchmark through the common ENM-pocket predictor."""

import argparse
from pathlib import Path

import predict


ROOT = Path(__file__).resolve().parent
TARGETS = {
    "kras": ("4OBE.cif", "A:1-169", (11, 12, 13, 14, 15, 16, 17, 18)),
    "bcr": (
        "1OPL.cif", "A:242-533",
        (267, 268, 272, 275, 288, 289, 290, 305, 309, 318, 332,
         334, 335, 336, 337, 338, 340, 389, 399, 400, 401),
    ),
    "myosin": ("5TBY.cif", "A:6-780", (184,)),
    "mapk14": ("1R39.cif", "A:4-351", (*range(30, 39), 53, 168)),
    "ptpn1": ("1A5Y.cif", "A:2-285", (181, *range(216, 222), 262)),
    "pdk2": (
        "2BTZ.cif", "A:6-384",
        (*range(251, 259), 290, *range(325, 331)),
    ),
    "ptpn11": ("4DGP.cif", "A:3-528", (425, *range(459, 466), 506)),
    "cmyc": (
        "1NKP.cif", ("A:897-984", "B:202-284"),
        (898, 902, 903, 905, 906, 907, 909, 910, 911, 913, 914, 918,
         937, 938, 939, 940),
    ),
}

# Targets whose domain spans more than one chain give a tuple of windows.
# c-Myc is the Myc/Max dimer: one copy of each chain, DNA excluded from the
# node set, with the DNA contact face of Myc as the orthosteric surface.


def run_target(target, *, pair_weight=0.1, replay=False,
               out_root=ROOT / "results"):
    structure, residue_range, orthosteric = TARGETS[target]
    windows = ((residue_range,) if isinstance(residue_range, str)
               else tuple(residue_range))
    arguments = [
        str(ROOT / "data" / target / structure),
        "--out-dir", str(Path(out_root) / target),
        "--pair-weight", str(pair_weight),
    ]
    for window in windows:
        arguments.extend(("--chain", window.split(":", 1)[0], "--range", window))
    for number in orthosteric:
        arguments.extend(("--orthosteric", f"A:{number}"))
    if replay:
        arguments.extend(("--replay", str(ROOT / "results" / target)))
    predict.main(arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", *TARGETS))
    parser.add_argument("--pair-weight", type=float, default=0.1)
    parser.add_argument("--replay", action="store_true", help="reuse the included pocket")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for target in TARGETS if args.target == "all" else (args.target,):
        print(f"[{target.upper()}]")
        run_target(target, pair_weight=args.pair_weight,
                   replay=args.replay, out_root=args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
