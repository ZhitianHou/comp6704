"""Post-hoc robustness experiments; original formal records remain untouched."""
from study_common import ROOT, np, synthetic, solve, certificate, threadpool_limits, threadpool_info
from study_io import resolve_run, write_csv
from datetime import datetime,timezone
import csv
import hashlib
import json
import platform
import shutil


def settings(config):
    cases=[]
    for c in config['correlations']:
        for a in config['algorithms']:
            for freq in config['diagnostic_frequencies']:
                cases.append((c,a,config['rho_frequency_study'],freq))
    for rho in config['rho_high_correlation']:
        case=(.9,'ADMM',rho,config['rho_study_check_every'])
        if case not in cases:
            cases.append(case)
    return cases


def main():
    config=json.loads((ROOT/'ablation_protocol.json').read_text())
    cases=settings(config)
    assert len(cases)==config['expected_unique_settings']
    assert len(cases)*len(config['data_seeds'])*config['timing_repetitions']==config['expected_timed_solves']
    source=resolve_run()
    frozen=json.loads((source/'config.json').read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/f'ablations_{stamp}'
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    for name in ['solvers.py','study_common.py','study_io.py','run_ablations.py','ablation_protocol.json']:
        shutil.copy2(ROOT/name,out/'source'/name)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    problem_rows=list(csv.DictReader((source/'problems.csv').open()))
    records=[]; coefficients={}
    with threadpool_limits(limits=1):
        Xw,yw,*_=synthetic(1000,200,.5,101)
        lam=.05*np.max(np.abs(Xw.T@yw/len(yw)))
        for a in config['algorithms']:
            solve(Xw,yw,lam,a)
        # Scheduled checks and the final cap must always retain a valid certificate.
        for a in config['algorithms']:
            _,h,s=solve(Xw,yw,lam,a,max_iter=7,check_every=10,tol=1e-14)
            assert [v['iteration'] for v in h]==[0,7]
            assert not s['converged'] and h[-1]['iteration']==7
        with (out/'history.jsonl').open('w') as history:
            for seed in config['data_seeds']:
                data={}
                for c in config['correlations']:
                    job=f'synthetic_c{c}_s{seed}'
                    p=next(v for v in problem_rows if v['job']==job)
                    syn=frozen['synthetic']
                    X,y,*_=synthetic(syn['n_train'],syn['p'],c,seed,n_test=syn['n_test'],sparsity=syn['sparsity'],noise_sd=syn['noise_sd'])
                    data[c]=(X,y,float(p['lam']))
                for rep in range(config['timing_repetitions']):
                    order=np.random.default_rng(20261120+100*seed+rep).permutation(len(cases))
                    for idx in order:
                        c,a,rho,freq=cases[idx];X,y,lam=data[c]
                        w,h,s=solve(X,y,lam,a,rho=rho,check_every=freq,
                                    tol=config['relative_gap_tol'],max_iter=config['max_iter'])
                        assert s['converged'] and h[-1]['relative_gap']<=config['relative_gap_tol']
                        cert=certificate(X,y,w,lam)
                        # CD works on a Fortran-layout copy. BLAS reduction order
                        # can differ by roundoff from this independent C-layout check.
                        assert np.isclose(cert['objective'],h[-1]['objective'],rtol=1e-12,atol=1e-12)
                        assert cert['relative_gap']<=config['relative_gap_tol']+1e-12
                        run=f'c{c}_s{seed}_{a}_rho{rho}_q{freq}_r{rep}'
                        records.append(dict(run=run,correlation=c,seed=seed,algorithm=a,rho=rho,
                                            repeat=rep,**s,**cert))
                        coefficients[run]=w
                        for item in h:
                            history.write(json.dumps(dict(run=run,**item))+'\n')
                write_csv(out/'raw_runs.csv',records)
                print(f'seed {seed}: {len(records)}/210 solves complete',flush=True)
        np.savez_compressed(out/'coefficients.npz',**coefficients)
        write_csv(out/'raw_runs.csv',records)
        meta=dict(stage=config['stage'],formal_source=source.name,all_converged=True,
                  timed_solves=len(records),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').iterdir()},
                  threadpools=threadpool_info(),platform=platform.platform(),numpy=np.__version__,
                  max_relative_gap=max(r['relative_gap'] for r in records))
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    (ROOT/'latest_ablations.txt').write_text(str(out)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
