"""(1) prospectivity skill: random CV vs block CV vs truth in a new area; (2) deposits found with a fixed hole budget.
python evaluate.py [dev|heldout] [out.json]"""
import json, sys, time
import numpy as np
from basin import sim, model, drill


def run(which):
    if which == "heldout":
        h = json.load(open("heldout/config.json"))
        seeds, cfg, budget = h["seeds"], h["cfg"], h["budget"]
    else:
        seeds, cfg, budget = list(range(1, 13)), None, 40
    V, D = {}, {p: dict(found=0, total=0, first=[]) for p in ("grid", "greedy", "bayes")}
    picks = dict(random_cv=0, block_cv=0, truth=0)
    for s in seeds:
        b = sim.make(s, cfg)
        e = model.evaluate(b, s)
        for m, v in e.items():
            for k, x in v.items():
                V.setdefault(f"{m}/{k}", []).append(float(x))
        better = "logistic" if e["logistic"]["truth_new_area"] >= e["knn_xy"]["truth_new_area"] else "knn_xy"
        for k in ("random_cv", "block_cv"):
            picks[k] += (max(e, key=lambda m: e[m][k]) == better)
        pr = model.prospectivity(b)
        for p in D:
            r = drill.run(b, pr, p, budget)
            D[p]["found"] += r["found"]; D[p]["total"] += r["total"]; D[p]["first"].append(r["first"] or budget + 1)
    return dict(set=which, worlds=len(seeds), budget=budget, auc={k: float(np.nanmean(v)) for k, v in V.items()},
                picks_better_model=picks, drilling={p: dict(found=d["found"], total=d["total"], median_first=float(np.median(d["first"])),
                                                          no_discovery=int(sum(f > budget for f in d["first"]))) for p, d in D.items()})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    t0 = time.time()
    r = run(which)
    print(f"{which} ({time.time() - t0:.0f}s): {r['worlds']} basins")
    for m in ("logistic", "knn_xy"):
        a = r["auc"]
        print(f"  {m:9s} AUC  random CV {a[m + '/random_cv']:.3f}   block CV {a[m + '/block_cv']:.3f}   truth in new area {a[m + '/truth_new_area']:.3f}")
    print(f"  picks the model that is really better in the new area: random CV {r['picks_better_model']['random_cv']}/{r['worlds']}, "
          f"block CV {r['picks_better_model']['block_cv']}/{r['worlds']}")
    for p, d in r["drilling"].items():
        print(f"  drilling {p:7s} deposits found {d['found']:3d}/{d['total']} with {r['budget']} holes per basin   median holes to first {d['median_first']:.0f}   basins with none {d['no_discovery']}")
    if len(sys.argv) > 2:
        json.dump(r, open(sys.argv[2], "w"), indent=1)
