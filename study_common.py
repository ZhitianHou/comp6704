"""Shared data generation, preprocessing and numerical validation for the study."""
import os
from pathlib import Path

# Apply the recorded native-thread request before importing numerical libraries.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-lasso-mpl")
for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits, threadpool_info
from solvers import solve, certificate, soft

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ["ISTA", "CD", "ADMM"]


def preprocess(X, y, train, test, beta=None):
    mean, scale = X[train].mean(0), X[train].std(0)
    scale[scale == 0] = 1.
    ymean = y[train].mean()
    return ((X[train]-mean)/scale, y[train]-ymean,
            (X[test]-mean)/scale, y[test]-ymean,
            None if beta is None else beta*scale)


def synthetic(n, p, correlation, seed, *, n_test=None, sparsity=None, noise_sd=.5):
    n_test = n if n_test is None else n_test
    sparsity = max(3, p//10) if sparsity is None else sparsity
    if n <= 0 or n_test <= 0 or p <= 0 or not 1 <= sparsity <= p or not 0 <= correlation < 1 or noise_sd < 0:
        raise ValueError('Invalid synthetic-data dimensions, sparsity, correlation or noise')
    rng = np.random.default_rng(seed)
    # Same latent draws, support, coefficients, and noise across correlation levels.
    Z = rng.normal(size=(n+n_test, p))
    X = Z.copy()
    for j in range(1, p):
        X[:, j] = correlation*X[:, j-1] + np.sqrt(1-correlation**2)*Z[:, j]
    beta = np.zeros(p)
    support = rng.choice(p, sparsity, replace=False)
    beta[support] = rng.choice([-1., 1.], size=len(support))
    y = X @ beta + noise_sd*rng.normal(size=n+n_test)
    return preprocess(X, y, np.arange(n), np.arange(n, n+n_test), beta)


def correctness():
    records = []
    # Orthonormal design has a closed-form solution, independent of sklearn.
    X = np.sqrt(8)*np.eye(8)
    beta = np.array([-2., -1., -.1, 0., .1, .3, 1., 2.])
    y = X @ beta
    target = soft(beta, .2)
    for name in ALGORITHMS:
        w, _, stats = solve(X, y, .2, name, tol=1e-12)
        error = float(np.max(np.abs(w-target)))
        assert error < 1e-5, (name, error)
        records.append(dict(check="orthonormal_closed_form", algorithm=name, max_error=error))
        zero, _, _ = solve(X, y, 2.1, name, tol=1e-12)
        assert np.array_equal(zero, np.zeros(8))
    # Correlated, nontrivial case cross-checked against an external implementation.
    X, y, _, _, _ = synthetic(100, 20, .8, 101)
    lam = .05*np.max(np.abs(X.T @ y/len(y)))
    reference = Lasso(alpha=lam, fit_intercept=False, tol=1e-13, max_iter=100000).fit(X, y)
    ref_cert = certificate(X, y, reference.coef_, lam)
    assert ref_cert["relative_gap"] < 1e-10
    for name in ALGORITHMS:
        w, hist, stats = solve(X, y, lam, name, tol=1e-10)
        delta = abs(hist[-1]["objective"]-ref_cert["objective"])
        assert stats["converged"] and delta < 1e-7
        assert np.max(np.abs(w-reference.coef_)) < 1e-3
        if name in ("ISTA", "CD"):
            assert np.max(np.diff([h["objective"] for h in hist])) <= 1e-10
        records.append(dict(check="sklearn_correlated", algorithm=name, objective_error=delta,
                            max_coefficient_error=float(np.max(np.abs(w-reference.coef_)))))
    return records
