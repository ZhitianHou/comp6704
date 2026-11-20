"""Audit saved arrays, certificates, split reconstruction and source hashes; no solves."""
from study_common import ROOT, np, certificate, synthetic, preprocess, load_diabetes, threadpool_limits
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import argparse
from study_io import resolve_run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    source=resolve_run(args.run)
    out=ROOT/'outputs'/source.name
    rows=list(csv.DictReader((source/'raw_runs.csv').open()))
    meta=json.loads((source/'metadata.json').read_text())
    assert len(rows)==162
    assert len({r['job'] for r in rows})==18
    assert set(Counter((r['job'],r['algorithm']) for r in rows).values())=={3}
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    present=[(source/'data'/f'{job}.npz').exists() for job in {r['job'] for r in rows}]
    if any(present) and not all(present):
        raise ValueError('Only some processed arrays are present; restore the complete run before a full audit')
    arrays_present=all(present)
    audited=0
    with threadpool_limits(limits=1):
        for job in sorted({r['job'] for r in rows}):
            subset=[r for r in rows if r['job']==job]
            r=subset[0];seed=int(r['seed'])
            if r['dataset']=='synthetic':
                X,y,Xt,yt,truth=synthetic(1000,200,float(r['correlation']),seed)
            else:
                rawx,rawy=load_diabetes(return_X_y=True,scaled=False)
                split=json.loads((source/'data'/f'{job}_split.json').read_text())
                all_indices=split['train']+split['validation']+split['test']
                assert sorted(all_indices)==list(range(442))
                X,y,Xt,yt,truth=preprocess(rawx,rawy,split['train'],split['test'])
            assert np.max(np.abs(X.mean(0)))<1e-12
            assert np.max(np.abs(X.std(0)-1))<1e-12
            if arrays_present:
                data=np.load(source/'data'/f'{job}.npz')
                for key,expected in [('Xtrain',X),('ytrain',y),('Xtest',Xt),('ytest',yt)]:
                    np.testing.assert_array_equal(data[key],expected)
                ref=certificate(X,y,data['reference'],float(r['lam']))
                assert ref['relative_gap']<1e-10
            for record in subset:
                assert record['converged']=='True'
                assert float(record['relative_gap'])<=1e-6
                if arrays_present:
                    w=data[f"{record['algorithm']}_repeat{record['repeat']}"]
                    cert=certificate(X,y,w,float(record['lam']))
                    for key in ['objective','gap','relative_gap','kkt_scaled']:
                        assert np.isclose(cert[key],float(record[key]),rtol=1e-10,atol=1e-12)
                    mse=float(np.mean((Xt@w-yt)**2))
                    assert np.isclose(mse,float(record['test_mse']),rtol=1e-12,atol=1e-12)
                audited+=1
    result=dict(passed=True,rows=audited,source_hashes_verified=True,
                processed_data_regenerated_and_checked=True,
                persisted_arrays_and_certificates_checked=arrays_present,
                note='The repository excludes processed npz arrays; full coefficient checks require regenerated or locally retained arrays.')
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
