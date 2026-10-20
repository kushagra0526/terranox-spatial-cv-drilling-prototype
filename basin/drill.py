"""Sequential drilling in the new (east) area with a fixed hole budget.

A hole returns ORE (within the ore footprint), ALTERATION (inside a halo, missed the ore) or BARREN.
  grid       evenly spaced holes over the area
  greedy     static: highest prospectivity first, never updated (skips cells within 3 of a drilled hole)
  bayes      (this) per-cell deposit-centre probabilities from prospectivity; after each hole, Bayes-update every cell by
             the outcome's likelihood given a centre there (barren rules out centres within the halo radius; alteration
             raises centres at 2..halo cells and rules out the hole's immediate neighbourhood); next hole = max P(ore)
"""
import numpy as np

EPS_OTHER = 0.08             # chance a hole sees alteration from some other deposit


def outcome(b, y, x):
    if b["ore"][y, x]:
        return "ORE"
    return "ALTERATION" if b["halo"][y, x] else "BARREN"


def run(b, prosp, policy, budget=40):
    n, c = b["n"], b["cfg"]
    H, F = c["halo"], c["foot"]
    yy, xx = b["yy"], b["xx"]
    area = xx >= n // 2
    found, holes, first = set(), [], None
    dep = b["deposits"]
    if policy == "grid":
        k = int(np.ceil(np.sqrt(budget / 2)))                             # area is twice as tall as wide
        ys = np.linspace(4, n - 5, 2 * k).astype(int); xs = np.linspace(n // 2 + 4, n - 5, k).astype(int)
        cells = [(y, x) for y in ys for x in xs][:budget]
    p = np.where(area, prosp, 0.0)
    p = p / p.sum() * c["deposits"] / 2 * 0.05                          # prior probability a deposit centre is at a cell
    p = np.clip(p, 1e-6, 0.5)
    drilled = np.zeros((n, n), bool)
    for t in range(budget):
        if policy == "grid":
            y, x = cells[t]
        else:
            score = p if policy == "bayes" else np.where(area, prosp, -1.0)
            if policy == "bayes":
                score = blur_sum(p, F)
            score = np.where(drilled | ~area, -1, score)
            if policy == "greedy":
                block = np.zeros_like(drilled)
                for (a, bb) in holes:
                    block |= (yy - a) ** 2 + (xx - bb) ** 2 <= 9
                score = np.where(block, -1, score)
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        drilled[y, x] = True
        holes.append((y, x))
        o = outcome(b, y, x)
        if o == "ORE":
            k = int(np.argmin((dep[:, 0] - y) ** 2 + (dep[:, 1] - x) ** 2))
            if k not in found:
                found.add(k)
                if first is None:
                    first = t + 1
        if policy == "bayes":
            d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
            if o == "BARREN":
                like1 = np.where(d <= H, 0.0, 1.0)                        # a centre within the halo would have shown
                like0 = np.full_like(p, 1 - EPS_OTHER)
            elif o == "ALTERATION":
                like1 = np.where(d <= F, 0.0, np.where(d <= H, 1.0, EPS_OTHER))
                like0 = np.full_like(p, EPS_OTHER)
            else:
                like1 = np.where(d <= F, 1.0, EPS_OTHER)
                like0 = np.full_like(p, EPS_OTHER)
            p = p * like1 / (p * like1 + (1 - p) * like0)
            if o == "ORE":                                                # this deposit is found: its halo holds no other centre
                p = np.where(d <= 2 * H, 1e-6, p)
            p = np.clip(p, 1e-6, 0.999)
    return dict(found=len(found), first=first, holes=holes, total=int(((dep[:, 1]) >= n // 2).sum()))


def blur_sum(p, r):
    """P(ore at cell) ~ sum of centre probabilities within the ore footprint radius."""
    out = p.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dy or dx) and dy * dy + dx * dx <= r * r:
                out += np.roll(np.roll(p, dy, 0), dx, 1)
    return out
