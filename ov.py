
import json, csv, pathlib
out = {}
for t in ["kras","bcr","myosin","mapk14","ptpn1","pdk2","ptpn11","cmyc"]:
    m = json.loads(pathlib.Path(f"results/{t}/method.json").read_text())
    rows = list(csv.DictReader(open(f"results/{t}/residue_scores.csv")))
    hits = [r["residue_id"] for r in csv.DictReader(open(f"results/{t}/hit_list.csv"))]
    out[t] = dict(
        cavity=sorted(m["pocket"]["selected_pocket"]),
        ranking=m["pocket"]["pocket_ranking"],
        families=len(m["pocket"].get("detected_ensemble_pockets", [])),
        top5=hits,
        energy=m.get("qubo", {}).get("energy"),
        seed=m["orthosteric_candidates"],
        nodes=len(rows),
        score={r["residue_id"]: float(r["quantum_score"]) for r in rows},
    )
pathlib.Path("/tmp/pipe_overlay.json").write_text(json.dumps(out, separators=(",",":")))
print("bytes", len(json.dumps(out)))
