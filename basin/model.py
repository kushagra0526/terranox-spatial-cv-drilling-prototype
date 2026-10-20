"""Prospectivity models and how their skill is measured.

Label for a historical hole: did it hit alteration (inside a halo)? Two models:
  logistic   on survey features only
  knn_xy     k-nearest holes on (x, y, features) -- the "memorise the neighbourhood" kind that random CV loves
Skill: AUC under random 5-fold CV, spatial block CV (20x20-cell blocks held out together), and the truth: every cell of
the unexplored east half scored against the real halo map.
"""
import numpy as np


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def standardise(X, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) + 1e-9 if sd is None else sd
    return (X - mu) / sd, mu, sd


def fit_logistic(X, y, l2=1.0, iters=300, lr=0.2):
    Xs, mu, sd = standardise(X)
    Xb = np.c_[np.ones(len(Xs)), Xs]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g * 4
    return lambda Z: 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(Z)), standardise(Z, mu, sd)[0]] @ w, -30, 30)))


def fit_knn_xy(X, y, xy, k=5, w_xy=3.0):
    Xs, mu, sd = standardise(X)
    P = np.c_[xy / 10.0 * w_xy, Xs]

    def pred(Z, zxy):
        Q = np.c_[zxy / 10.0 * w_xy, standardise(Z, mu, sd)[0]]
        out = np.empty(len(Q))
        for i0 in range(0, len(Q), 2000):
            d = ((Q[i0:i0 + 2000, None, :] - P[None]) ** 2).sum(-1)
            nn = np.argsort(d, axis=1)[:, :k]
            out[i0:i0 + 2000] = y[nn].mean(1) + 1e-3 * (-d[np.arange(len(nn))[:, None], nn].mean(1))
        return out
    return pred


def holes_xy(b):
    return b["hist"].astype(float)


def data(b):
    h = b["hist"]
    return b["feat"][h[:, 0], h[:, 1]], b["halo"][h[:, 0], h[:, 1]].astype(float), holes_xy(b)


def cv_auc(b, model, folds):
    X, y, xy = data(b)
    sc = np.empty(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if model == "logistic":
            sc[te] = fit_logistic(X[tr], y[tr])(X[te])
        else:
            sc[te] = fit_knn_xy(X[tr], y[tr], xy[tr])(X[te], xy[te])
    return auc(sc, y)


def evaluate(b, seed=0):
    r = np.random.default_rng(seed)
    X, y, xy = data(b)
    rnd = r.permutation(len(y)) % 5
    blocks = (xy[:, 0] // 20) * 100 + (xy[:, 1] // 20)
    ub = np.unique(blocks); bf = dict(zip(ub, r.permutation(len(ub)) % 5))
    blk = np.array([bf[v] for v in blocks])
    n = b["n"]
    east = np.argwhere(b["xx"] >= n // 2)
    Xe, ye = b["feat"][east[:, 0], east[:, 1]], b["halo"][east[:, 0], east[:, 1]]
    out = {}
    for m in ("logistic", "knn_xy"):
        if m == "logistic":
            pe = fit_logistic(X, y)(Xe)
        else:
            pe = fit_knn_xy(X, y, xy)(Xe, east.astype(float))
        out[m] = dict(random_cv=cv_auc(b, m, rnd), block_cv=cv_auc(b, m, blk), truth_new_area=auc(pe, ye))
    return out


def prospectivity(b):
    """Logistic model trained on all historical holes, scored on every cell."""
    X, y, _ = data(b)
    f = fit_logistic(X, y)
    n = b["n"]
    return f(b["feat"].reshape(-1, b["feat"].shape[-1])).reshape(n, n)
