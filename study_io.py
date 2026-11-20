"""Shared paths and serialization for local and relocated study deliveries."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_run(value=None):
    value = Path(value) if value is not None else Path((ROOT/'latest_formal.txt').read_text().strip())
    candidate = value if value.is_absolute() else ROOT/value
    if candidate.is_dir():
        return candidate.resolve()
    relocated = ROOT/'results'/value.name
    if relocated.is_dir():
        return relocated.resolve()
    raise FileNotFoundError(f'Run directory does not exist: {candidate}')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
