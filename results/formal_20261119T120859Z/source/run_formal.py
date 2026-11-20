"""Frozen formal experiment; raw results append to a fresh run directory."""
from run_pilot import (ROOT, ALGORITHMS, synthetic, preprocess, correctness,
                       solve, certificate, Lasso, load_diabetes, np,
                       threadpool_limits, threadpool_info)
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import os
import time
from datetime import datetime, timezone
import scipy
import sklearn


def write_csv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as f:
        writer=csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=__import__('pathlib').Path)
    args=parser.parse_args()
    config=json.loads((ROOT/"formal_protocol.json").read_text())
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=args.output or ROOT/"results"/f"formal_{stamp}"
    out.mkdir(parents=True,exist_ok=False)
    (out/"data").mkdir(); (out/"source").mkdir()
    for name in ["formal_protocol.json","solvers.py","run_pilot.py","run_formal.py","requirements-lock.txt"]:
        shutil.copy2(ROOT/name,out/"source"/name)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    all_start=time.perf_counter()
    records=[]
    problems=[]
    with threadpool_limits(limits=1):
        checks=correctness()
        (out/"correctness.json").write_text(json.dumps(checks,indent=2))
        # Untimed, separate-data warmup; no results from it enter the study.
        Xw,yw,*_=synthetic(1000,200,.5,101)
        warm_lam=.05*np.max(np.abs(Xw.T@yw/len(yw)))
        warmups=[]
        for algo in ALGORITHMS:
            _,_,s=solve(Xw,yw,warm_lam,algo)
            assert s['converged']
            warmups.append(dict(algorithm=algo,**s))
        (out/"warmup.json").write_text(json.dumps(warmups,indent=2))
        jobs=[]
        syn=config['synthetic']
        for c in syn['correlations']:
            for seed in syn['seeds']:
                jobs.append((f"synthetic_c{c}_s{seed}","synthetic",c,seed,
                             synthetic(syn['n_train'],syn['p'],c,seed),None))
        Xraw,yraw=load_diabetes(return_X_y=True,scaled=False)
        for seed in config['diabetes']['split_seeds']:
            indices=np.random.default_rng(seed).permutation(len(yraw))
            a,b=int(.7*len(yraw)),int(.85*len(yraw))
            train,test=indices[:a],indices[b:]
            split=dict(train=train.tolist(),validation=indices[a:b].tolist(),test=test.tolist(),
                       feature_mean=Xraw[train].mean(0).tolist(),
                       feature_scale=Xraw[train].std(0).tolist(),target_mean=float(yraw[train].mean()))
            jobs.append((f"diabetes_s{seed}","diabetes",None,seed,
                         preprocess(Xraw,yraw,train,test),split))
        history_file=(out/"history.jsonl").open("w")
        for jid,dataset,c,seed,data,split in jobs:
            X,y,Xtest,ytest,truth=data
            n,p=X.shape
            lam=config['lambda_fraction']*np.max(np.abs(X.T@y/n))
            model=Lasso(alpha=lam,fit_intercept=False,tol=1e-13,max_iter=100000).fit(X,y)
            ref=certificate(X,y,model.coef_,lam)
            assert ref['relative_gap']<1e-10,(jid,ref)
            arrays=dict(Xtrain=X,ytrain=y,Xtest=Xtest,ytest=ytest,reference=model.coef_)
            if truth is not None:
                arrays['truth']=truth
            if split:
                (out/"data"/f"{jid}_split.json").write_text(json.dumps(split,indent=2))
            cond=float(np.linalg.cond(X))
            corrmat=X.T@X/n
            corrmean=float(np.mean(np.abs(corrmat[np.triu_indices(p,1)])))
            problem=dict(job=jid,dataset=dataset,correlation=c,seed=seed,n_train=n,p=p,
                         n_test=len(ytest),lam=float(lam),design_condition=cond,
                         mean_absolute_correlation=corrmean,reference_objective=ref['objective'],
                         reference_relative_gap=ref['relative_gap'],
                         reference_test_mse=float(np.mean((Xtest@model.coef_-ytest)**2)))
            if truth is not None:
                actual=truth!=0; selected=np.abs(model.coef_)>config['support_threshold']
                problem.update(reference_coefficient_relative_error=float(np.linalg.norm(model.coef_-truth)/np.linalg.norm(truth)),
                    reference_support_f1=float(2*np.count_nonzero(selected&actual)/(np.count_nonzero(selected)+np.count_nonzero(actual))))
            problems.append(problem)
            for repeat in range(config['timing_repetitions']):
                order=np.random.default_rng(10000+100*seed+repeat+len(jid)).permutation(ALGORITHMS)
                for order_index,algo in enumerate(order):
                    w,h,stats=solve(X,y,lam,str(algo),tol=config['relative_gap_tol'],
                                    max_iter=config['max_iter'],rho=config['admm_rho'])
                    cert=h[-1]
                    assert cert['objective']>=ref['objective']-1e-8*max(1.,ref['objective'])
                    if algo in ['ISTA','CD']:
                        assert np.max(np.diff([v['objective'] for v in h]))<=1e-9*max(1.,h[0]['objective'])
                    row=dict(**problem,algorithm=str(algo),repeat=repeat,order_index=order_index,
                             **stats,**cert,time_to_tolerance_s=stats['total_s'] if stats['converged'] else None,
                             diagnostic_fraction=stats['diagnostic_s']/stats['total_s'],
                             objective_delta=cert['objective']-ref['objective'],
                             max_coefficient_difference=float(np.max(np.abs(w-model.coef_))),
                             test_mse=float(np.mean((Xtest@w-ytest)**2)),
                             coefficient_relative_error=None,support_f1=None)
                    if truth is not None:
                        selected=np.abs(w)>config['support_threshold']; actual=truth!=0
                        row.update(coefficient_relative_error=float(np.linalg.norm(w-truth)/np.linalg.norm(truth)),
                            support_f1=float(2*np.count_nonzero(selected&actual)/(np.count_nonzero(selected)+np.count_nonzero(actual))))
                    records.append(row)
                    arrays[f"{algo}_repeat{repeat}"]=w
                    run=f"{jid}_{algo}_r{repeat}"
                    for item in h:
                        history_file.write(json.dumps(dict(run=run,**item))+"\n")
            np.savez_compressed(out/"data"/f"{jid}.npz",**arrays)
            history_file.flush()
            print(f"{jid}: completed {len(records)}/162 timed solves; all converged={all(r['converged'] for r in records)}",flush=True)
        history_file.close()
        write_csv(out/"raw_runs.csv",records)
        write_csv(out/"problems.csv",problems)
        assert len(records)==config['expected_timed_solves']
        assert len(problems)==18
        # Exact repeats must agree numerically; timing repeats are not new datasets.
        for jid,*_ in jobs:
            for algo in ALGORITHMS:
                subset=[r for r in records if r['job']==jid and r['algorithm']==algo]
                assert len({r['iterations'] for r in subset})==1
                assert len({r['objective'] for r in subset})==1
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (out/"source").iterdir()}
        metadata=dict(created_utc=stamp,python=sys.version,platform=platform.platform(),
                      numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                      threadpools=threadpool_info(),native_thread_environment={v:os.environ[v] for v in
                        ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
                      thread_note="Single native thread requested; Apple Accelerate not fully enumerated.",
                      source_sha256=hashes,all_converged=all(r['converged'] for r in records),
                      timed_solves=len(records),problems=len(problems),
                      correctness_and_repeat_checks_passed=True,total_wall_s=time.perf_counter()-all_start,
                      max_relative_gap=max(r['relative_gap'] for r in records),
                      max_objective_delta=max(r['objective_delta'] for r in records))
        (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    (ROOT/"latest_formal.txt").write_text(str(out)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
