"""Build a compact local handoff; do not publish or upload anything."""
from pathlib import Path
import hashlib
import json
import zipfile
import argparse

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-only',action='store_true',help='Create a local source/results handoff without the reading PDF')
    args=parser.parse_args()
    stamp=Path((ROOT/'latest_formal.txt').read_text().strip()).name
    entries={}
    for name in ['README.txt','research_plan.html','solvers.py','run_pilot.py','run_formal.py',
                 'export_formal.py','verify_formal.py','package_delivery.py','check_delivery.py',
                 'study_io.py','review_checks.py','render_pilot_report.py','CODE_REVIEW.txt',
                 'export_review_analysis.py',
                 'run_ablations.py','export_ablations.py','verify_ablations.py','ablation_protocol.json',
                 'data_superconductivity.py','run_realdata.py','export_realdata.py','verify_realdata.py','realdata_protocol.json',
                 'formal_protocol.json','requirements-lock.txt']:
        entries[name]=(ROOT/name).read_bytes()
    entries['latest_formal.txt']=('results/'+stamp+'\n').encode()
    ablation=Path((ROOT/'latest_ablations.txt').read_text().strip()).name
    entries['latest_ablations.txt']=('results/'+ablation+'\n').encode()
    realdata=Path((ROOT/'latest_realdata.txt').read_text().strip()).name
    entries['latest_realdata.txt']=('results/'+realdata+'\n').encode()
    for name in ['README.md','.gitignore']:
        if (ROOT/name).exists():
            entries[name]=(ROOT/name).read_bytes()
    for folder in [ROOT/'results'/stamp,ROOT/'outputs'/stamp,ROOT/'results'/ablation,ROOT/'outputs'/ablation,ROOT/'results'/realdata,ROOT/'outputs'/realdata]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix!='.npz' and p.name!='raw_sha256.json':
                entries[str(p.relative_to(ROOT))]=p.read_bytes()
    for p in (ROOT/'report').iterdir():
        if p.suffix in ['.tex','.sty']:
            entries[str(p.relative_to(ROOT))]=p.read_bytes()
    for name in ['review_checks.json','review_spectral.json']:
        p=ROOT/'output'/name
        if p.exists():
            entries[str(p.relative_to(ROOT))]=p.read_bytes()
    if not args.code_only:
        report=Path((ROOT/'latest_report.txt').read_text().strip())
        entries[str(report)]=(ROOT/report).read_bytes()
        entries['latest_report.txt']=(str(report)+'\n').encode()
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    entries['PACKAGE_SHA256.json']=json.dumps(manifest,indent=2).encode()
    path=ROOT/'output'/('lasso_code_and_results.zip' if args.code_only else 'lasso_reproducibility.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(entries.items()):
            z.writestr('lasso_study/'+name,data)
    print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,files=len(entries))))


if __name__=='__main__':
    main()
