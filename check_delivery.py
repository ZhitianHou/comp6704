"""Check archive hashes, portable export and TeX compilation; never solve again."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
import re

ROOT=Path(__file__).resolve().parent


def main():
    archive=ROOT/'output'/'lasso_reproducibility.zip'
    stamp=Path((ROOT/'latest_formal.txt').read_text().strip()).name
    with tempfile.TemporaryDirectory(prefix='comp6704-lasso-delivery-') as temp:
        with zipfile.ZipFile(archive) as z:
            assert not any('/.venv/' in n or '/__pycache__/' in n or n.endswith('.npz') for n in z.namelist())
            z.extractall(temp)
        root=Path(temp)/'lasso_study'
        manifest=json.loads((root/'PACKAGE_SHA256.json').read_text())
        for name,digest in manifest.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        subprocess.run([sys.executable,'export_formal.py'],cwd=root,check=True,capture_output=True,text=True)
        for name in ['per_problem.csv','aggregate.csv','model_recovery.csv','timing_table.tex','recovery_table.tex']:
            assert (root/'outputs'/stamp/name).read_bytes()==(ROOT/'outputs'/stamp/name).read_bytes(),name
        subprocess.run([sys.executable,'export_ablations.py'],cwd=root,check=True,capture_output=True,text=True)
        ablation=Path((ROOT/'latest_ablations.txt').read_text().strip()).name
        for name in ['aggregate.csv','per_problem.csv','frequency_pairs.csv','ablation_table.tex']:
            assert (root/'outputs'/ablation/name).read_bytes()==(ROOT/'outputs'/ablation/name).read_bytes(),name
        subprocess.run([sys.executable,'export_realdata.py'],cwd=root,check=True,capture_output=True,text=True)
        realdata=Path((ROOT/'latest_realdata.txt').read_text().strip()).name
        for name in ['aggregate.csv','per_problem.csv','realdata_table.tex']:
            assert (root/'outputs'/realdata/name).read_bytes()==(ROOT/'outputs'/realdata/name).read_bytes(),name
        for _ in range(2):
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                           cwd=root/'report',check=True,capture_output=True,text=True)
        log=(root/'report'/'main.log').read_text()
        assert 'undefined' not in log and 'Overfull' not in log
        match=re.search(r'Output written on main.pdf \((\d+) pages',log)
        assert match and 1<=int(match.group(1))<=10
        result=dict(passed=True,archive_bytes=archive.stat().st_size,package_hashes_checked=len(manifest),
                    isolated_export_matches=True,isolated_tex_compile_pages=int(match.group(1)),
                    timed_solves_repeated=0,environment_and_processed_arrays_excluded=True)
    (ROOT/'output'/'delivery_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
