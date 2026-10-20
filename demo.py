"""One basin: how good the prospectivity model looks under each validation, and 30 holes under each drilling policy."""
from basin import sim, model, drill

cfg = dict(conductors=12, deposits=10, halo=5, em_noise=0.6, grav_noise=0.8, missing_faults=2, hist_cluster=0.75)
b = sim.make(107, cfg)                       # held-out basin 107; the README table averages all 12
e = model.evaluate(b, 107)
print("Prospectivity skill (AUC), trained on the clustered historical holes in the west:")
for m, v in e.items():
    print(f"  {m:9s} random CV {v['random_cv']:.2f}   spatial block CV {v['block_cv']:.2f}   truth in the new east area {v['truth_new_area']:.2f}")
pr = model.prospectivity(b)
print("\n30 holes in the east area:")
for p in ("grid", "greedy", "bayes"):
    r = drill.run(b, pr, p, 30)
    seq = "".join({"ORE": "O", "ALTERATION": "a", "BARREN": "."}[drill.outcome(b, y, x)] for y, x in r["holes"])
    print(f"  {p:7s} deposits found {r['found']}/{r['total']}   first at hole {r['first'] or '-'}   {seq}")
print("\n  O = ore, a = alteration (missed the ore but inside its halo), . = barren")
