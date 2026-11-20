"""Additional real regression benchmark, with a frozen independent protocol."""
from study_common import ROOT, np, solve, certificate, Lasso, threadpool_limits, threadpool_info
from study_io import write_csv
from data_superconductivity import load_data,prepare_split
from datetime import datetime,timezone
from time import perf_counter
import hashlib
import json
import platform
import shutil
import warnings
from sklearn.exceptions import ConvergenceWarning


def main():
    Xraw,yraw,group,names,config=load_data()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/f'realdata_{stamp}';out.mkdir(exist_ok=False)
    (out/'source').mkdir();(out/'splits').mkdir()
    for name in ['solvers.py','study_common.py','study_io.py','data_superconductivity.py','run_realdata.py','realdata_protocol.json','requirements-lock.txt']:
        shutil.copy2(ROOT/name,out/'source'/name)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'features.json').write_text(json.dumps(names,indent=2))
    rows=[];problems=[];arrays={};start=perf_counter()
    with threadpool_limits(limits=1):
        # Warm numerical kernels on separate data, capped at 50 updates; not a pilot search.
        data,_=prepare_split(Xraw,yraw,group,config['warmup_seed'],config)
        X,y,*_=data;lam=config['lambda_fraction']*np.max(np.abs(X.T@y/len(y)))
        for algo in config['algorithms']:
            solve(X,y,lam,algo,max_iter=50,tol=config['relative_gap_tol'],rho=config['rho'])
        with (out/'history.jsonl').open('w') as history:
            for seed in config['split_seeds']:
                data,details=prepare_split(Xraw,yraw,group,seed,config)
                X,y,Xt,yt=data;n,p=X.shape
                (out/'splits'/f'seed{seed}.json').write_text(json.dumps(details,indent=2))
                lam=config['lambda_fraction']*np.max(np.abs(X.T@y/n))
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    reference=Lasso(alpha=lam,fit_intercept=False,tol=1e-13,max_iter=100000).fit(X,y)
                ref=certificate(X,y,reference.coef_,lam)
                if ref['relative_gap']>1e-10:
                    raise RuntimeError('External reference did not reach the required accuracy')
                mse_ref=float(np.mean((Xt@reference.coef_-yt)**2))
                problem=dict(seed=seed,n_train=n,n_validation=details['n_validation'],n_test=len(yt),p=p,
                             lam=float(lam),design_condition=float(np.linalg.cond(X)),reference_objective=ref['objective'],
                             reference_relative_gap=ref['relative_gap'],reference_test_mse=mse_ref,
                             reference_test_r2=1-mse_ref/float(np.var(yt)))
                problems.append(problem);arrays[f's{seed}_reference']=reference.coef_
                for repeat in range(config['timing_repetitions']):
                    for algo in np.random.default_rng(3000+100*seed+repeat).permutation(config['algorithms']):
                        w,h,s=solve(X,y,lam,str(algo),tol=config['relative_gap_tol'],max_iter=config['max_iter'],
                                    rho=config['rho'],check_every=config['check_every'])
                        mse=float(np.mean((Xt@w-yt)**2));rid=f's{seed}_{algo}_r{repeat}'
                        rows.append(dict(run=rid,algorithm=str(algo),repeat=repeat,**problem,**s,**h[-1],
                                         time_to_tolerance_s=s['total_s'] if s['converged'] else None,
                                         test_mse=mse,test_r2=1-mse/float(np.var(yt)),
                                         objective_delta=h[-1]['objective']-ref['objective']))
                        arrays[rid]=w
                        for item in h:
                            history.write(json.dumps(dict(run=rid,**item))+'\n')
                        print(f'{rid}: {s["iterations"]} iterations, {s["total_s"]:.3f}s, gap={h[-1]["relative_gap"]:.2e}, converged={s["converged"]}',flush=True)
                write_csv(out/'raw_runs.csv',rows);write_csv(out/'problems.csv',problems)
                history.flush()
        np.savez_compressed(out/'coefficients.npz',**arrays)
        assert len(rows)==config['expected_timed_solves']
        meta=dict(stage='additional_real_dataset',dataset='UCI Superconductivity',timed_solves=len(rows),
                  all_converged=all(r['converged'] for r in rows),max_relative_gap=max(r['relative_gap'] for r in rows),
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').iterdir()},
                  raw_archive_sha256=config['zip_sha256'],raw_csv_sha256=config['csv_sha256'],
                  feature_groups=len(np.unique(group)),rows=len(yraw),features=Xraw.shape[1],
                  platform=platform.platform(),numpy=np.__version__,threadpools=threadpool_info(),
                  native_threads_requested=1,wall_s=perf_counter()-start)
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    (ROOT/'latest_realdata.txt').write_text(str(out)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
