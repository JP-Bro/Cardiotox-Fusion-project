"""
reproduce_check.py -- clean-rebuild test (reviewer: "the committed CSVs must be produced by the committed code").

Copies the repo, deletes the derived data files, re-runs scripts/03_rebuild_pipeline.py with a different
PYTHONHASHSEED, and requires the rebuilt cohort, expression matrix and both splits to equal the committed ones.
Needs data/raw (LINCS files).  Run:  python scripts/reproduce_check.py
"""
import os, sys, shutil, subprocess
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sandbox import make_copy, ROOT

FILES = ['data/processed/lincs_matched_compounds.csv', 'data/processed/expression_matrix.csv',
         'data/splits/drug_split.csv', 'data/splits/scaffold_split.csv']
tmp, dst = make_copy()
try:
    for f in FILES:
        os.remove(os.path.join(dst, f))
    env = dict(os.environ, PYTHONHASHSEED='12345')
    r = subprocess.run([sys.executable, os.path.join(dst, 'scripts', '03_rebuild_pipeline.py')], cwd=dst, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:]); sys.exit(1)
    bad = []
    for f in FILES:
        a = pd.read_csv(os.path.join(ROOT, f)); b = pd.read_csv(os.path.join(dst, f))
        same = a.shape == b.shape and list(a.columns) == list(b.columns) and a.equals(b) or \
               (a.shape == b.shape and all(a[c].equals(b[c]) or (a[c].dtype.kind == 'f' and abs(a[c] - b[c]).max() < 1e-9) for c in a.columns))
        print(f"{'SAME' if same else 'DIFFERENT'}  {f}  {a.shape}")
        if not same: bad.append(f)
    print("\nREPRODUCIBLE from a clean rebuild" if not bad else f"\nNOT reproducible: {bad}")
    sys.exit(1 if bad else 0)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
