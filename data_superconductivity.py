"""Pinned official UCI data and label-independent, duplicate-aware splitting."""
from study_common import ROOT, np
import hashlib
import io
import json
import urllib.request
import zipfile


def load_data(download=False):
    config=json.loads((ROOT/'realdata_protocol.json').read_text())
    path=ROOT/'data/raw/superconductivity/uci_464.zip'
    if not path.exists():
        if not download:
            raise FileNotFoundError('Run data_superconductivity.py --download to retrieve the pinned official data')
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(config['url'],timeout=60) as response:
            payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=config['zip_sha256']:
            raise ValueError('Downloaded archive checksum mismatch')
        path.write_bytes(payload)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['zip_sha256']:
        raise ValueError('Archive checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv=z.read(config['file'])
    if hashlib.sha256(csv).hexdigest()!=config['csv_sha256']:
        raise ValueError('CSV checksum mismatch')
    header=[v.strip('"') for v in csv.splitlines()[0].decode().split(',')]
    values=np.loadtxt(io.BytesIO(csv),delimiter=',',skiprows=1)
    if list(values.shape)!=config['raw_shape'] or not np.isfinite(values).all() or header[-1]!=config['target']:
        raise ValueError('Unexpected dataset schema or missing values')
    X,y=values[:,:-1],values[:,-1]
    _,group=np.unique(X,axis=0,return_inverse=True)
    if len(np.unique(group))!=config['expected_feature_groups']:
        raise ValueError('Unexpected duplicate-feature grouping')
    return X,y,group,header[:-1],config


def prepare_split(X,y,group,seed,config):
    groups=np.random.default_rng(seed).permutation(np.unique(group))
    a=int(config['train_group_fraction']*len(groups));b=int(config['validation_end_group_fraction']*len(groups))
    partitions=[np.flatnonzero(np.isin(group,g)) for g in [groups[:a],groups[a:b],groups[b:]]]
    train,val,test=partitions
    mean=X[train].mean(0);scale=X[train].std(0);ymean=float(y[train].mean())
    if np.any(scale==0):
        raise ValueError('Zero-variance training feature: explicitly revise preprocessing before fitting')
    data=((X[train]-mean)/scale,y[train]-ymean,(X[test]-mean)/scale,y[test]-ymean)
    details=dict(seed=seed,train=train.tolist(),validation=val.tolist(),test=test.tolist(),
                 feature_mean=mean.tolist(),feature_scale=scale.tolist(),target_mean=ymean,
                 group_count=len(groups),n_train=len(train),n_validation=len(val),n_test=len(test))
    return data,details


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    X,y,g,_,_=load_data(args.download)
    print(json.dumps(dict(rows=len(y),features=X.shape[1],feature_groups=len(np.unique(g)))))
