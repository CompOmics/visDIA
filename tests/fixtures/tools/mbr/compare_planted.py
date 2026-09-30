import json, sys
a = json.load(open(r"C:/Users/robbi/mumdia_viewer_ref/smoke/planted.json"))
b = json.load(open(r"C:/Users/robbi/AppData/Local/Temp/claude/C--Users-robbi-OneDrive-Documents-GitHub-visDIA/ecfc3918-c866-4e97-9df8-df3b2807a87f/scratchpad/tmp/G3_mbr/planted_b.json"))
print("options a:", {k: a["options"][k] for k in ("n_planted","windows","seed","cycles","cycle_seconds")})
print("options b:", {k: b["options"][k] for k in ("n_planted","windows","seed","cycles","cycle_seconds")})
print("windows a:", [(round(w["lower"],4), round(w["upper"],4)) for w in a["isolation_windows"]])
print("windows b:", [(round(w["lower"],4), round(w["upper"],4)) for w in b["isolation_windows"]])
same = all(abs(x["lower"]-y["lower"])<1e-9 and abs(x["upper"]-y["upper"])<1e-9 for x,y in zip(a["isolation_windows"], b["isolation_windows"]))
print("windows identical:", same)
pa_ = {p["candidate_id"]: p for p in a["planted"]}
pb_ = {p["candidate_id"]: p for p in b["planted"]}
shared = sorted(set(pa_) & set(pb_))
print("planted a:", len(pa_), "planted b:", len(pb_), "shared:", len(shared), "only a:", len(set(pa_)-set(pb_)), "only b:", len(set(pb_)-set(pa_)))
import statistics
d = [pb_[c]["apex_seconds"] - pa_[c]["apex_seconds"] for c in shared]
if d:
    print("apex_b - apex_a over shared: min %.2f median %.2f max %.2f" % (min(d), statistics.median(d), max(d)))
# which only-a precursors fall in b's windows (so could be extracted in b)?
blo = b["isolation_windows"][0]["lower"]; bhi = b["isolation_windows"][-1]["upper"]
print("b m/z span", blo, bhi)
only_a = [c for c in pa_ if c not in pb_]
print("only-a inside b span:", sum(blo <= pa_[c]["precursor_mz"] < bhi for c in only_a))
only_b = [c for c in pb_ if c not in pa_]
alo = a["isolation_windows"][0]["lower"]; ahi = a["isolation_windows"][-1]["upper"]
print("a m/z span", alo, ahi)
print("only-b inside a span:", sum(alo <= pb_[c]["precursor_mz"] < ahi for c in only_b))
json.dump({"shared": shared, "only_a": sorted(only_a), "only_b": sorted(only_b)}, open(r"C:/Users/robbi/AppData/Local/Temp/claude/C--Users-robbi-OneDrive-Documents-GitHub-visDIA/ecfc3918-c866-4e97-9df8-df3b2807a87f/scratchpad/tmp/G3_mbr/planted_sets.json","w"))
