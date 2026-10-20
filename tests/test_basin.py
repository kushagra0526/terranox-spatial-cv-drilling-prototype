import numpy as np
from basin import sim, model, drill


def test_auc():
    assert model.auc(np.array([0.1, 0.2, 0.8, 0.9]), [0, 0, 1, 1]) == 1.0
    assert model.auc(np.array([0.9, 0.8, 0.2, 0.1]), [0, 0, 1, 1]) == 0.0


def test_deposits_sit_on_conductors_inside_halos():
    b = sim.make(2)
    for y, x in b["deposits"]:
        assert b["ore"][y, x] and b["halo"][y, x]


def test_outcomes():
    b = sim.make(3)
    y, x = b["deposits"][0]
    assert drill.outcome(b, y, x) == "ORE"
    assert drill.outcome(b, y, min(x + 3, b["n"] - 1)) in ("ALTERATION", "ORE")


def test_random_cv_overrates_coordinate_knn():
    b = sim.make(4)
    e = model.evaluate(b, 4)
    assert e["knn_xy"]["random_cv"] > e["knn_xy"]["block_cv"] + 0.15


def test_bayes_update_rules_out_barren_neighbourhood():
    b = sim.make(5)
    pr = model.prospectivity(b)
    r = drill.run(b, pr, "bayes", budget=5)
    assert len(r["holes"]) == 5 and len(set(r["holes"])) == 5


def test_bayes_finds_at_least_as_many_as_greedy_on_dev_seeds():
    tot = {"greedy": 0, "bayes": 0}
    for s in (6, 7):
        b = sim.make(s); pr = model.prospectivity(b)
        for p in tot:
            tot[p] += drill.run(b, pr, p, budget=30)["found"]
    assert tot["bayes"] >= tot["greedy"]
