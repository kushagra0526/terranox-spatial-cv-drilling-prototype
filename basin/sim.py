"""A synthetic Athabasca-style basin with known deposits.

Grid cells are 250 m. Deposits sit on graphitic basement conductors, mostly near fault intersections, inside alteration
halos much larger than the ore. Surveys see the conductors (EM), a gravity low over the halo, and weak noisy radiometrics.
West half = explored historically (clustered holes near known deposits); east half = the new project area.
"""
import numpy as np

DEV = dict(n=160, conductors=8, faults=6, deposits=16, halo=6, foot=1, em_noise=0.35, grav_noise=0.5, rad_noise=1.0,
           hist_holes=400, hist_cluster=0.6, missing_faults=1, grav_amp=0.8, rad_amp=0.3)


def _line(r, n, length, step_sd):
    x, y = r.uniform(0, n), r.uniform(0, n)
    ang = r.uniform(0, np.pi)
    pts = []
    for _ in range(length):
        ang += r.normal(0, step_sd)
        x += np.cos(ang); y += np.sin(ang)
        if 0 <= x < n and 0 <= y < n:
            pts.append((int(y), int(x)))
    return pts


def blur(a, k):
    from numpy.fft import rfft2, irfft2
    n = a.shape[0]
    yy, xx = np.mgrid[:n, :n]
    d2 = np.minimum(yy, n - yy) ** 2 + np.minimum(xx, n - xx) ** 2
    g = np.exp(-d2 / (2 * k * k)); g /= g.sum()
    return irfft2(rfft2(a) * rfft2(g), s=a.shape)


def dist_to(mask):
    """Approximate distance (cells) to the nearest True cell (two-pass chamfer)."""
    n = mask.shape[0]
    d = np.where(mask, 0.0, 1e9)
    for i in range(n):
        for j in range(n):
            if i: d[i, j] = min(d[i, j], d[i - 1, j] + 1)
            if j: d[i, j] = min(d[i, j], d[i, j - 1] + 1)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i < n - 1: d[i, j] = min(d[i, j], d[i + 1, j] + 1)
            if j < n - 1: d[i, j] = min(d[i, j], d[i, j + 1] + 1)
    return d


def make(seed, cfg=None):
    c = {**DEV, **(cfg or {})}
    r = np.random.default_rng(seed)
    n = c["n"]
    cond = np.zeros((n, n), bool)
    for _ in range(c["conductors"]):
        for p in _line(r, n, 220, 0.06):
            cond[p] = True
    fault = np.zeros((n, n), bool)
    faults = []
    for _ in range(c["faults"]):
        pts = _line(r, n, 260, 0.0)
        faults.append(pts)
        for p in pts:
            fault[p] = True
    dc, df = dist_to(cond), dist_to(fault)
    yy, xx = np.mgrid[:n, :n]
    # deposits: on conductors, 75% within 4 cells of a fault
    cand_int = np.argwhere(cond & (df <= 4))
    cand_any = np.argwhere(cond)
    deps = []
    while len(deps) < c["deposits"]:
        pool = cand_int if (r.random() < 0.75 and len(cand_int)) else cand_any
        y, x = pool[r.integers(len(pool))]
        if all((y - a) ** 2 + (x - b) ** 2 > (2 * c["halo"]) ** 2 for a, b in deps):
            deps.append((int(y), int(x)))
    dep = np.array(deps)
    dmin = np.min(np.sqrt((yy[..., None] - dep[:, 0]) ** 2 + (xx[..., None] - dep[:, 1]) ** 2), axis=2)
    ore, halo = dmin <= c["foot"], dmin <= c["halo"]
    em = blur(cond.astype(float), 1.5) * 6 + r.normal(0, c["em_noise"], (n, n))
    grav = -blur(halo.astype(float), 2.0) * c["grav_amp"] + blur(r.normal(0, 1, (n, n)), 6) * 4 + r.normal(0, c["grav_noise"], (n, n))
    rad = blur(halo.astype(float), 1.0) * c["rad_amp"] + r.normal(0, c["rad_noise"], (n, n))
    mapped = np.zeros((n, n), bool)                                        # some faults were never mapped
    for k, pts in enumerate(faults):
        if k >= c["missing_faults"]:
            for p in pts:
                mapped[p] = True
    picks = em > 3.0                                                       # conductor picks from the EM survey
    feat = np.stack([em, grav, rad, np.exp(-dist_to(picks) / 3), np.exp(-dist_to(mapped) / 4)], -1)
    feat = np.concatenate([feat, (feat[..., 3] * feat[..., 4])[..., None]], -1)
    # historical holes in the west half, clustered near deposits found there
    west = xx < n // 2
    hist = []
    known = [d for d in deps if d[1] < n // 2]
    while len(hist) < c["hist_holes"]:
        if known and r.random() < c["hist_cluster"]:
            y, x = known[r.integers(len(known))]
            y, x = int(np.clip(y + r.normal(0, 5), 0, n - 1)), int(np.clip(x + r.normal(0, 5), 0, n // 2 - 1))
        else:
            pool = np.argwhere(picks & west)
            y, x = pool[r.integers(len(pool))] if len(pool) else (r.integers(n), r.integers(n // 2))
        hist.append((int(y), int(x)))
    hist = np.array(hist)
    return dict(cfg=c, n=n, deposits=dep, ore=ore, halo=halo, feat=feat, hist=hist, xx=xx, yy=yy)
