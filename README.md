# Honest prospectivity validation and halo-aware drilling (synthetic uranium basin)

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Terranox AI.

Terranox trains uranium prospectivity models on 70+ years of exploration outcomes. It uses "sequential decision-making: choosing the next exploration action to maximize information per dollar spent, updating from each result" ([YC](https://www.ycombinator.com/companies/terranox-ai)). Two things decide whether that works, and both can be measured in a simulation where the deposits are known:
1. **How model skill is measured.** Historical holes cluster around known deposits, so random cross-validation scores a hole using its neighbours. Spatial block CV ([Roberts et al. 2017](https://doi.org/10.1111/ecog.02881)) asks the question that matters: how well does the model rank ground nobody has drilled?
2. **What a hole teaches.** In Athabasca-style unconformity systems the alteration halo is far bigger than the ore. A hole that misses but hits alteration says "close"; a barren hole rules out a whole neighbourhood. A policy that drills the top of a static map ignores both.

```
$ python demo.py   (held-out basin 107)
  logistic  random CV 0.79   spatial block CV 0.78   truth in the new east area 0.77
  knn_xy    random CV 0.95   spatial block CV 0.58   truth in the new east area 0.66
30 holes in the east area:
  grid    deposits found 0/4   ..............................
  greedy  deposits found 0/4   .a..a..aa.a....a....a.........
  bayes   deposits found 4/4   O..aaaaaaaaa.aOaOO............
  O = ore, a = alteration (missed the ore but inside its halo), . = barren
```

## What it does (`basin/`, numpy)
- **`sim.py`: a 40 × 40 km basin (250 m cells).**
  - Graphitic conductors and faults (some unmapped).
  - Deposits on conductors, mostly near fault intersections, each with a 1-cell ore footprint and a 5–6 cell alteration halo.
  - Surveys: EM conductor picks, gravity (halo low + regional field + noise), noisy radiometrics.
  - Historical holes in the west half, clustered near deposits found there; the east half is the new project area.
- **`model.py`:** logistic prospectivity on survey features, and a coordinate-kNN model, each scored by random 5-fold CV, spatial block CV (5 km blocks), and the truth over every cell of the new area.
- **`drill.py`:** a fixed hole budget in the new area.
  - **grid**;
  - **greedy:** top of the static prospectivity map;
  - **bayes (this):** per-cell deposit-centre probabilities from the map. After every hole they are Bayes-updated by the outcome: barren rules out centres within the halo, alteration raises centres 2–halo cells away and rules out the hole's spot, ore removes that deposit's halo. The next hole is max P(ore).

## Results
**Held-out:** [`heldout/config.json`](heldout/config.json), frozen in git (commit `3cbf0a1`) before the first run. A different basin type: more decoy conductors, fewer deposits, smaller halos, noisier surveys, two unmapped faults, more clustered history, and a 30-hole budget. 12 basins, first run.

| model | random CV AUC | spatial block CV AUC | **truth in the new area** |
|---|---|---|---|
| logistic (survey features) | 0.823 | 0.677 | **0.684** |
| kNN on coordinates + features | 0.932 | 0.551 | 0.629 |

Random CV overstated skill by 0.14 (logistic) and 0.30 (kNN); block CV was within 0.01–0.08 of the truth. Picking the model that is really better in the new area: random CV got it right 4/12, block CV 7/12. The two models were closer in this basin type than on dev, where it was 1/12 vs 10/12.

| drilling, 30 holes per basin | deposits found | median holes to first | basins with no discovery |
|---|---|---|---|
| grid | 0 / 62 | — | 12 |
| greedy (static map) | 11 / 62 | 14 | 5 |
| **bayes, halo-aware (this)** | **37 / 62** | **3** | **0** |

Dev (12 basins, 40 holes): 72 / 99 vs greedy 22 / 99 vs grid 3 / 99.

## Honest limits
- **The drilling result is optimistic.** The Bayes policy's observation model (round halos, alteration always recognised in core) matches the simulator. Real halos are irregular and plunge with structure, and core logging misses alteration. A robust version needs a noisy observation model and halo shapes learned from past holes.
- **2-D, one hole = one cell.** No depth (unconformity depth varies 300–800 m in the real basin), no drill cost differences, no fences or step-outs.
- **The geology is a caricature** built from public descriptions. Nothing here uses Terranox data or reflects their models.

## Run
```
pip install numpy pytest
python demo.py
python evaluate.py heldout     # ~20 s; python -m pytest -q (6 tests)
```
