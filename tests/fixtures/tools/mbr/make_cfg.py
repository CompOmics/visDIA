"""Deep-merge SMOKE/exp_seq.json with an MBR overlay; write to the scratch dir."""
import json, sys
def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
    return a
base = json.load(open(r"C:/Users/robbi/mumdia_viewer_ref/smoke/exp_seq.json"))
PY = "C:/Users/robbi/AppData/Local/MuMDIA/python/Scripts/python.exe"
variants = {
    "cfg_mbr.json": {"mbr": {"strategy": "rt_transfer", "python": PY}},
    "cfg_mbr_a1.json": {"mbr": {"strategy": "rt_transfer", "python": PY, "min_anchor_runs": 1}},
}
for name, overlay in variants.items():
    c = merge(json.loads(json.dumps(base)), overlay)
    json.dump(c, open(name, "w"), indent=2)
    print(name, json.dumps(c))
