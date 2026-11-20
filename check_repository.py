"""Verify file hashes, portable result export and paper compilation without training."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-tex', action='store_true', help='Check saved results without compiling the paper')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    hashes_checked = 0
    for name, digest in manifest.items():
        # Exports regenerate these files (PDF metadata can change without a
        # numerical change). Tables are checked independently below.
        if name.startswith('outputs/') or (name.startswith('output/') and name.endswith('.json')):
            continue
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Published file has changed: {name}')
        hashes_checked += 1

    exports = {
        'formal': ['per_problem.csv', 'aggregate.csv', 'model_recovery.csv', 'timing_table.tex', 'recovery_table.tex'],
        'ablations': ['aggregate.csv', 'per_problem.csv', 'frequency_pairs.csv', 'ablation_table.tex'],
        'realdata': ['aggregate.csv', 'per_problem.csv', 'realdata_table.tex'],
    }
    with tempfile.TemporaryDirectory(prefix='lasso-repository-') as temp:
        root = Path(temp)
        for name in manifest:
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        for study, tables in exports.items():
            subprocess.run([sys.executable, f'export_{study}.py'], cwd=root,
                           check=True, capture_output=True, text=True)
            stamp = Path((ROOT/f'latest_{study}.txt').read_text().strip()).name
            for name in tables:
                relative = Path('outputs')/stamp/name
                if (root/relative).read_bytes() != (ROOT/relative).read_bytes():
                    raise ValueError(f'Regenerated table differs: {relative}')
        subprocess.run([sys.executable, 'export_review_analysis.py'], cwd=root,
                       check=True, capture_output=True, text=True)
        stamp = Path((ROOT/'latest_formal.txt').read_text().strip()).name
        spectral = Path('outputs')/stamp/'spectral_table.tex'
        if (root/spectral).read_bytes() != (ROOT/spectral).read_bytes():
            raise ValueError(f'Regenerated table differs: {spectral}')

        pages = None
        if not args.skip_tex:
            for _ in range(2):
                subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                               cwd=root/'report', check=True, capture_output=True, text=True)
            log = (root/'report'/'main.log').read_text()
            match = re.search(r'Output written on main.pdf \((\d+) pages', log)
            if 'undefined' in log or 'Overfull' in log or not match or not 1 <= int(match.group(1)) <= 10:
                raise ValueError('Paper compilation has unresolved references, overflow or an invalid page count')
            pages = int(match.group(1))
            if (root/'report'/'main.pdf').stat().st_size >= 20_000_000:
                raise ValueError('Compiled report exceeds the submission size limit')
    print(json.dumps(dict(passed=True, hashes_checked=hashes_checked, isolated_exports_match=True,
                          compiled_pages=pages, timed_solves_repeated=0), indent=2))


if __name__ == '__main__':
    main()
