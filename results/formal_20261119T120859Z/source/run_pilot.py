"""Run correctness checks and a bounded pilot; never launches the formal study."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-lasso-mpl")
for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
import argparse
import csv
import hashlib
import json
import platform
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def synthetic(n, p, correlation, seed):
    rng = np.random.default_rng(seed)
    # Same latent draws, support, coefficients, and noise across correlation levels.
    Z = rng.normal(size=(2*n, p))
    X = Z.copy()
    for j in range(1, p):
        X[:, j] = correlation*X[:, j-1] + np.sqrt(1-correlation**2)*Z[:, j]
    beta = np.zeros(p)
    support = rng.choice(p, max(3, p//10), replace=False)
    beta[support] = rng.choice([-1., 1.], size=len(support))
    y = X @ beta + .5*rng.normal(size=2*n)
    return preprocess(X, y, np.arange(n), np.arange(n, 2*n), beta)


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


def plot(rows, histories, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    colors = dict(ISTA="#377eb8", CD="#e07a26", ADMM="#2b995b")
    for col, corr in enumerate([0., .5, .9]):
        for name in ALGORITHMS:
            key = f"synthetic_300x60_c{corr}_s11_{name}"
            h = histories[key]
            gap = np.maximum([v["relative_gap"] for v in h], 1e-14)
            axes[0,col].semilogy([v["iteration"] for v in h], gap, label=name, color=colors[name])
            axes[1,col].semilogy([1000*v["elapsed_s"] for v in h], gap, color=colors[name])
        axes[0,col].set_title(f"AR(1) correlation = {corr}")
        axes[0,col].set_xlabel("Outer iteration (CD: full sweep)")
        axes[1,col].set_xlabel("Milliseconds, including setup and diagnostics")
        axes[1,col].xaxis.set_major_locator(MaxNLocator(4))
        for ax in axes[:,col]:
            ax.axhline(1e-6, color="gray", linestyle="--", linewidth=.8)
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Duality gap / F(0)")
    axes[1,0].set_ylabel("Duality gap / F(0)")
    axes[0,0].legend()
    fig.suptitle("LASSO pilot: one timing per run; seed 11; not final results")
    fig.savefig(out/"pilot_convergence.png", dpi=170)
    fig.savefig(out/"pilot_convergence.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or ROOT/"results"/f"pilot_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    config = dict(stage="pilot", seed_list=[11,12], correlations=[0.,.5,.9],
                  n_train=300, n_test=300, p=60, sparsity=6, noise_sd=.5,
                  lambda_fraction=.05, rho=1., relative_gap_tol=1e-6,
                  max_iter=20000, blas_threads=1, diagnostic_every=1,
                  timing_repetitions=1, data_dtype="float64")
    (out/"config.json").write_text(json.dumps(config, indent=2))
    begin = perf_counter()
    with threadpool_limits(limits=1):
        checks = correctness()
        (out/"correctness.json").write_text(json.dumps(checks, indent=2))
        jobs = []
        for corr in config["correlations"]:
            for seed in config["seed_list"]:
                jobs.append((f"synthetic_300x60_c{corr}_s{seed}", "synthetic", corr, seed,
                             synthetic(300,60,corr,seed)))
        jobs.append(("scale_1000x200_c0.9_s11", "scale_probe", .9, 11, synthetic(1000,200,.9,11)))
        X,y = load_diabetes(return_X_y=True, scaled=False)
        indices = np.random.default_rng(11).permutation(len(y))
        # Train 70%, validation 15% reserved, test remainder. No tuning in pilot.
        a,b = int(.7*len(y)), int(.85*len(y))
        (out/"diabetes_split.json").write_text(json.dumps(dict(train=indices[:a].tolist(),
            validation=indices[a:b].tolist(), test=indices[b:].tolist())))
        jobs.append(("diabetes_s11", "diabetes", None, 11,
                     preprocess(X,y,indices[:a],indices[b:])))
        rows, histories = [], {}
        arrays = {}
        for job_id, dataset, corr, seed, data in jobs:
            X,y,Xtest,ytest,truth = data
            lam = config["lambda_fraction"]*np.max(np.abs(X.T @ y/len(y)))
            reference = Lasso(alpha=lam, fit_intercept=False, tol=1e-13, max_iter=100000).fit(X,y)
            ref = certificate(X,y,reference.coef_,lam)
            assert ref["relative_gap"] < 1e-10
            # Preserve exact processed arrays, not only generation parameters.
            for label, values in zip(["Xtrain","ytrain","Xtest","ytest"], data[:4]):
                arrays[f"{job_id}_{label}"] = values
            if truth is not None:
                arrays[f"{job_id}_truth"] = truth
            arrays[f"{job_id}_reference"] = reference.coef_
            condition = float(np.linalg.cond(X))
            corr_matrix = X.T@X/len(y)
            mean_abs_corr = float(np.mean(np.abs(corr_matrix[np.triu_indices(X.shape[1],1)])))
            order = np.random.default_rng(seed+len(job_id)).permutation(ALGORITHMS)
            for name in order:
                w,h,stats = solve(X,y,lam,str(name),tol=config["relative_gap_tol"],
                                  max_iter=config["max_iter"],rho=config["rho"])
                assert h[-1]["objective"] >= ref["objective"]-1e-8*max(1.,ref["objective"])
                row = dict(job=job_id,dataset=dataset,correlation=corr,seed=seed,algorithm=str(name),
                    n_train=len(y),p=X.shape[1],lam=float(lam),**stats,**h[-1],
                    design_condition=condition, mean_absolute_correlation=mean_abs_corr,
                    reference_objective=ref["objective"],reference_relative_gap=ref["relative_gap"],
                    objective_delta=h[-1]["objective"]-ref["objective"],
                    test_mse=float(np.mean((Xtest@w-ytest)**2)),
                    coefficient_relative_error=None if truth is None else float(np.linalg.norm(w-truth)/np.linalg.norm(truth)))
                if truth is not None:
                    selected=np.abs(w)>1e-6; actual=truth!=0
                    tp=np.count_nonzero(selected & actual)
                    row["support_f1"] = float(2*tp/max(1,np.count_nonzero(selected)+np.count_nonzero(actual)))
                else:
                    row["support_f1"] = None
                rows.append(row)
                histories[f"{job_id}_{name}"]=h
                arrays[f"{job_id}_{name}_coef"]=w
                print(f"{job_id:34s} {name:4s} k={stats['iterations']:5d} gap={h[-1]['relative_gap']:.2e} total={stats['total_s']:.4f}s",flush=True)
        np.savez_compressed(out/"arrays.npz",**arrays)
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (out/"summary.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        with (out/"history.jsonl").open("w") as f:
            for job, h in histories.items():
                for record in h:
                    f.write(json.dumps(dict(run=job,**record))+"\n")
        plot(rows,histories,out)
        for source in ["solvers.py","run_pilot.py"]:
            shutil.copy2(ROOT/source,out/source)
        meta=dict(created_utc=stamp,python=sys.version,platform=platform.platform(),
                  numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                  matplotlib=matplotlib.__version__,threadpools=threadpool_info(),
                  native_thread_environment={v:os.environ[v] for v in ["VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]},
                  thread_note="Requested one native thread; threadpoolctl may not enumerate Apple Accelerate.",
                  code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"solvers.py",ROOT/"run_pilot.py"]},
                  all_converged=all(r["converged"] for r in rows),runs=len(rows),
                  total_wall_s=perf_counter()-begin)
        (out/"metadata.json").write_text(json.dumps(meta,indent=2))
    (ROOT/"latest_pilot.txt").write_text(str(out)+"\n")
    print(json.dumps(meta,indent=2))


if __name__ == "__main__":
    main()
