"""Validate complete supplementary logs and, locally, persisted coefficients."""
from study_common import ROOT, np, synthetic, certificate, threadpool_limits
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json


def main():
    source=ROOT/'results'/Path((ROOT/'latest_ablations.txt').read_text().strip()).name
    rows=list(csv.DictReader((source/'raw_runs.csv').open()))
    config=json.loads((source/'config.json').read_text())
    meta=json.loads((source/'metadata.json').read_text())
    assert len(rows)==config['expected_timed_solves']==210
    assert len({r['run'] for r in rows})==210
    assert set(Counter((r['correlation'],r['seed'],r['algorithm'],r['rho'],r['check_every']) for r in rows).values())=={3}
    for name,sha in meta['source_sha256'].items():
        assert hashlib.sha256((source/'source'/name).read_bytes()).hexdigest()==sha
    data={}
    arrays=source/'coefficients.npz'
    coefs=np.load(arrays) if arrays.exists() else None
    with threadpool_limits(limits=1):
        for r in rows:
            assert r['converged']=='True' and float(r['relative_gap'])<=1e-6+1e-12
            if coefs is None:
                continue
            key=(float(r['correlation']),int(r['seed']))
            if key not in data:
                X,y,*_=synthetic(1000,200,*key)
                data[key]=(X,y,.05*np.max(np.abs(X.T@y/len(y))))
            X,y,lam=data[key]
            cert=certificate(X,y,coefs[r['run']],lam)
            for field in ['objective','relative_gap','kkt_scaled']:
                assert np.isclose(cert[field],float(r[field]),rtol=1e-10,atol=1e-12)
    counts=Counter(); last={}; previous=defaultdict(lambda:-1)
    byrun={r['run']:r for r in rows}
    for line in (source/'history.jsonl').open():
        record=json.loads(line);rid=record['run'];k=record['iteration']
        r=byrun[rid];q=int(r['check_every'])
        assert k>previous[rid]
        assert k%q==0 or k==config['max_iter']
        previous[rid]=k;counts[rid]+=1;last[rid]=record
    for rid,r in byrun.items():
        assert counts[rid]==int(r['certificate_checks'])
        assert last[rid]['iteration']==int(r['iterations'])
        assert last[rid]['relative_gap']<=1e-6
    result=dict(passed=True,timed_solves=len(rows),history_schedules_verified=True,
                source_hashes_verified=True,persisted_coefficients_checked=coefs is not None,
                posthoc_design=True,primary_logs_unchanged=True)
    out=ROOT/'output'/'validation'/f'{source.name}_audit.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
