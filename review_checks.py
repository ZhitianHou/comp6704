"""Independent numerical checks plus non-timing replay of frozen formal solutions."""
from study_common import ROOT, np, solve, threadpool_limits
from study_io import resolve_run
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from pathlib import Path
import csv
import json
import subprocess
import sys
import tempfile
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,help='Saved formal run; coefficient replay is skipped if processed arrays are absent')
    args=parser.parse_args()
    results=[]
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(20261120)
        # Unequal feature scales detect a missing CD denominator or wrong n scaling.
        X=rng.normal(size=(70,9))*np.geomspace(.3,3,9)
        y=X@np.array([1.,0.,-2.,0.,.4,0.,0.,0.,0.])+rng.normal(size=70)*.2
        lam=.12
        ref=Lasso(alpha=lam,fit_intercept=False,tol=1e-13,max_iter=100000).fit(X,y)
        n,p=X.shape
        # Independent bound-constrained QP: beta=a-b, a,b >= 0.
        def objective(v):
            w=v[:p]-v[p:]
            return np.sum((X@w-y)**2)/(2*n)+lam*np.sum(v)
        def gradient(v):
            g=X.T@(X@(v[:p]-v[p:])-y)/n
            return np.r_[g+lam,-g+lam]
        qp=minimize(objective,np.zeros(2*p),jac=gradient,bounds=[(0,None)]*(2*p),
                    method='SLSQP',options=dict(ftol=1e-12,maxiter=2000))
        if not qp.success:
            raise RuntimeError(qp.message)
        for algo in ['ISTA','CD','ADMM']:
            w,h,s=solve(X,y,lam,algo,tol=1e-10)
            assert s['converged']
            assert abs(h[-1]['objective']-qp.fun)<1e-8
            assert np.max(np.abs(w-ref.coef_))<1e-4
            residual=y-X@w
            theta=residual/n
            theta/=max(1.,np.max(np.abs(X.T@theta))/lam)
            assert np.max(np.abs(X.T@theta))<=lam*(1+1e-12)
            independent_gap=np.sum(residual**2)/(2*n)+lam*np.sum(np.abs(w))-y@theta+n/2*(theta@theta)
            assert abs(independent_gap-h[-1]['gap'])<1e-12
            results.append(dict(check='unequal_scale_sklearn_and_split_QP',algorithm=algo,objective_error=abs(h[-1]['objective']-qp.fun)))
        # Rank-deficient underdetermined design: compare fitted values/objectives,
        # since coefficient uniqueness is not assumed.
        X=rng.normal(size=(25,40));X[:,1]=X[:,0]
        y=X[:,0]-1.2*X[:,5]+.1*rng.normal(size=25)
        ref=Lasso(alpha=.15,fit_intercept=False,tol=1e-13,max_iter=100000).fit(X,y)
        target=np.sum((X@ref.coef_-y)**2)/50+.15*np.sum(np.abs(ref.coef_))
        for algo in ['ISTA','CD','ADMM']:
            w,h,s=solve(X,y,.15,algo,tol=1e-9)
            assert s['converged'] and abs(h[-1]['objective']-target)<1e-7
            results.append(dict(check='rank_deficient_underdetermined',algorithm=algo))
        for algo in ['ISTA','CD','ADMM']:
            _,h,s=solve(X,y,.15,algo,max_iter=0)
            assert not s['converged'] and s['iterations']==0 and len(h)==1
        X[:,2]=0
        for algo in ['ISTA','CD','ADMM']:
            w,h,s=solve(X,y,.15,algo,tol=1e-9)
            assert s['converged'] and w[2]==0
        results.append(dict(check='zero_column_and_iteration_cap',passed=True))
        invalid=[dict(lam=0),dict(lam=float('nan')),dict(tol=-1),dict(rho=float('inf')),dict(max_iter=-1)]
        for kwargs in invalid:
            params=dict(lam=.15);params.update(kwargs)
            try:
                solve(X,y,algorithm='ISTA',**params)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid parameter accepted: {kwargs}')
        try:
            solve(X,np.zeros((25,1)),.15,'ISTA')
        except ValueError:
            pass
        else:
            raise AssertionError('2D target accepted')
        source=resolve_run(args.run)
        rows=list(csv.DictReader((source/'raw_runs.csv').open()))
        replay_rows=[row for row in rows if row['repeat']=='0']
        available=[(source/'data'/f"{row['job']}.npz").exists() for row in replay_rows]
        if any(available) and not all(available):
            raise ValueError('Only some replay arrays are present; restore a complete run')
        for row in rows:
            if row['repeat']!='0' or not all(available):
                continue
            data=np.load(source/'data'/f"{row['job']}.npz")
            w,h,s=solve(data['Xtrain'],data['ytrain'],float(row['lam']),row['algorithm'])
            assert s['iterations']==int(row['iterations'])
            np.testing.assert_array_equal(w,data[f"{row['algorithm']}_repeat0"])
            assert h[-1]['objective']==float(row['objective'])
        results.append(dict(check='frozen_formal_numerical_replay',
                            status='passed' if all(available) else 'skipped',
                            problem_algorithm_combinations=len(replay_rows) if all(available) else 0,
                            coefficients_bitwise_identical=True if all(available) else None,
                            reason=None if all(available) else 'Processed arrays are excluded from the repository; rerun the formal study to regenerate them',
                            original_timing_records_untouched=True))
    with tempfile.TemporaryDirectory(prefix='lasso-review-') as tmp:
        p=Path(tmp)
        assert resolve_run(p)==p.resolve()
        (p/'metadata.json').write_text(json.dumps(dict(all_converged=False)))
        (p/'raw_runs.csv').write_text('converged\nFalse\n')
        (p/'problems.csv').write_text('job\nexample\n')
        check=subprocess.run([sys.executable,str(ROOT/'export_formal.py'),'--run',str(p)],capture_output=True,text=True)
        assert check.returncode!=0 and 'requires all runs to converge' in check.stderr
    results.append(dict(check='nonconvergence_and_custom_path',passed=True))
    out=ROOT/'output'/'review_checks.json'
    out.write_text(json.dumps(dict(passed=True,checks=results),indent=2))
    print(out)
    print(json.dumps(dict(passed=True,checks=results),indent=2))


if __name__=='__main__':
    main()
