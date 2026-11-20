"""Verify source, duplicate-group separation, preprocessing and saved certificates."""
from study_common import ROOT,np,certificate,threadpool_limits
from data_superconductivity import load_data,prepare_split
from pathlib import Path
import csv
import hashlib
import json


def main():
    source=ROOT/'results'/Path((ROOT/'latest_realdata.txt').read_text().strip()).name
    rows=list(csv.DictReader((source/'raw_runs.csv').open()))
    meta=json.loads((source/'metadata.json').read_text())
    assert len(rows)==27 and len({r['run'] for r in rows})==27
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((source/'source'/name).read_bytes()).hexdigest()==digest
    Xraw,yraw,group,_,config=load_data()
    arrays=np.load(source/'coefficients.npz') if (source/'coefficients.npz').exists() else None
    with threadpool_limits(limits=1):
        for seed in config['split_seeds']:
            (X,y,Xt,yt),split=prepare_split(Xraw,yraw,group,seed,config)
            saved=json.loads((source/'splits'/f'seed{seed}.json').read_text())
            assert saved==split
            groups=[set(group[saved[k]]) for k in ['train','validation','test']]
            assert all(groups[i].isdisjoint(groups[j]) for i,j in [(0,1),(0,2),(1,2)])
            assert sorted(saved['train']+saved['validation']+saved['test'])==list(range(len(yraw)))
            assert np.max(np.abs(X.mean(0)))<1e-11 and np.max(np.abs(X.std(0)-1))<1e-11
            selected=[r for r in rows if int(r['seed'])==seed]
            lam=.05*np.max(np.abs(X.T@y/len(y)))
            for r in selected:
                assert r['converged']=='True' and float(r['relative_gap'])<=1e-6
                if arrays is None:continue
                w=arrays[r['run']];cert=certificate(X,y,w,lam)
                assert np.isclose(cert['objective'],float(r['objective']),rtol=1e-12,atol=1e-10)
                assert np.isclose(cert['relative_gap'],float(r['relative_gap']),rtol=1e-6,atol=1e-12)
                assert np.isclose(np.mean((Xt@w-yt)**2),float(r['test_mse']),rtol=1e-12,atol=1e-10)
                assert cert['objective']>=float(r['reference_objective'])-1e-8
            if arrays is not None:
                ref=certificate(X,y,arrays[f's{seed}_reference'],lam)
                assert ref['relative_gap']<1e-10
    result=dict(passed=True,timed_solves=27,raw_checksums_verified=True,
                source_hashes_verified=True,split_reconstruction_verified=True,
                duplicate_feature_groups_disjoint=True,training_only_scaling_verified=True,
                persisted_coefficients_checked=arrays is not None)
    (ROOT/'outputs'/source.name/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
