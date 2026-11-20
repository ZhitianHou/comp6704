"""Dense educational LASSO solvers; objective ||Xw-y||²/(2n)+lam*||w||1."""
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def soft(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def certificate(X, y, w, lam):
    n = len(y)
    residual = y - X @ w
    correlations = X.T @ residual / n
    primal = residual @ residual / (2*n) + lam * np.abs(w).sum()
    theta = residual / n / max(1.0, np.max(np.abs(correlations)) / lam)
    dual = y @ theta - n/2 * (theta @ theta)
    raw_gap = float(primal-dual)
    if raw_gap < -1e-9 * max(1.0, primal):
        raise ArithmeticError("Invalid negative duality gap")
    active = w != 0
    violations = np.maximum(np.abs(correlations)-lam, 0.0)
    violations[active] = np.abs(correlations[active]-lam*np.sign(w[active]))
    baseline = y @ y / (2*n)
    lam_max = np.max(np.abs(X.T @ y / n))
    return dict(objective=float(primal), dual=float(dual), gap=max(0., raw_gap),
                relative_gap=max(0., raw_gap)/max(baseline, 1e-15),
                kkt_scaled=float(np.max(violations)/max(lam_max, 1e-15)))


def solve(X, y, lam, algorithm, tol=1e-6, max_iter=20000, rho=1.0):
    if lam <= 0 or rho <= 0:
        raise ValueError("lam and rho must be positive")
    n, p = X.shape
    start = perf_counter()
    w = np.zeros(p)
    if algorithm == "ISTA":
        # Exact spectral norm: preparation cost is included in timing.
        L = np.linalg.norm(X, 2)**2/n
        step = 1/L
    elif algorithm == "CD":
        X = np.asfortranarray(X)
        residual = y.copy()
        column_norms = np.sum(X*X, axis=0)/n
        if np.any(column_norms == 0):
            raise ValueError("Remove constant columns before fitting")
    elif algorithm == "ADMM":
        factor = cho_factor(X.T @ X/n + rho*np.eye(p))
        b = X.T @ y/n
        u = np.zeros(p)
    else:
        raise ValueError(algorithm)
    setup_s = perf_counter()-start
    diagnostic_s = 0.
    history = []
    update_s = 0.
    rnorm = snorm = None
    for k in range(max_iter+1):
        check_start = perf_counter()
        cert = certificate(X, y, w, lam)
        diagnostic_s += perf_counter()-check_start
        history.append(dict(iteration=k, elapsed_s=perf_counter()-start,
                            compute_s=setup_s+update_s, **cert,
                            primal_residual=rnorm, dual_residual=snorm))
        if cert["relative_gap"] <= tol or k == max_iter:
            break
        tick = perf_counter()
        if algorithm == "ISTA":
            w = soft(w + step*(X.T @ (y-X @ w))/n, step*lam)
        elif algorithm == "CD":
            for j in range(p):
                old = w[j]
                partial_dot = X[:, j] @ residual/n + column_norms[j]*old
                w[j] = float(soft(partial_dot, lam))/column_norms[j]
                residual -= X[:, j]*(w[j]-old)
        else:
            zold = w.copy()
            x = cho_solve(factor, b+rho*(w-u))
            w = soft(x+u, lam/rho)  # use feasible original-objective point z
            u += x-w
            rnorm = float(np.linalg.norm(x-w))
            snorm = float(rho*np.linalg.norm(w-zold))
        update_s += perf_counter()-tick
    return w, history, dict(converged=bool(cert["relative_gap"] <= tol),
                            iterations=k, setup_s=setup_s, update_s=update_s,
                            diagnostic_s=diagnostic_s, total_s=perf_counter()-start)
