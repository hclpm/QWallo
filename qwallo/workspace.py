"""Stage output layout and small IO helpers.

Every stage writes only into its own directory and reads only directories of
earlier stages, so a stage can be re-run alone and the dependency order is
visible in the filesystem.
"""

import csv
import json
from pathlib import Path

STAGES = {
    0: "00_structures", 1: "01_sites", 2: "02_network", 3: "03_propagation",
    4: "04_pockets", 5: "05_evaluation", 6: "06_baselines", 7: "07_robustness",
}


def stage_dir(work, target, stage, create=False):
    path = Path(work) / target / STAGES[stage]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def targets_from_argument(value, order):
    if value == "all":
        return list(order)
    if value not in order:
        raise SystemExit("unknown target: " + value)
    return [value]
