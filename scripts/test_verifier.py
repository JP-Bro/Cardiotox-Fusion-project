"""
test_verifier.py -- proves verify_pipeline.py can FAIL.

Reviewer rule: "before writing a check, ask what would make it fail".  This script copies the repo,
injects one realistic error at a time, runs the verifier on the copy, and requires it to exit non-zero
(and the named check to be among the failures).  A mutation the verifier does not catch is a hole.

Run:  python scripts/test_verifier.py [all | M1 M3 ...]
"""
import os, re, sys, shutil, subprocess
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sandbox import make_copy


def edit_csv(path, fn):
    df = pd.read_csv(path, index_col=0) if path.endswith('expression_matrix.csv') else pd.read_csv(path)
    df = fn(df)
    df.to_csv(path, index=path.endswith('expression_matrix.csv'))


def find(dst, name):
    for r, _, fs in os.walk(os.path.join(dst, 'data')):
        if 'raw' in r.split(os.sep):
            continue
        if name in fs:
            return os.path.join(r, name)
    raise FileNotFoundError(name)


def m_expr_value(dst):      # one expression value silently changed
    def f(d): d.iloc[3, 10] += 1.0; return d
    edit_csv(find(dst, 'expression_matrix.csv'), f)

def m_label_flip(dst):      # one label flipped in the matched file only
    def f(d): d.loc[5, 'cardiotox_label'] = 1 - d.loc[5, 'cardiotox_label']; return d
    edit_csv(find(dst, 'lincs_matched_compounds.csv'), f)

def m_dup_pert(dst):        # two compounds share one LINCS perturbagen
    def f(d): d.loc[1, 'pert_id'] = d.loc[0, 'pert_id']; return d
    edit_csv(find(dst, 'lincs_matched_compounds.csv'), f)

def m_drop_gene(dst):       # a landmark gene column lost
    def f(d): return d.drop(columns=d.columns[7])
    edit_csv(find(dst, 'expression_matrix.csv'), f)

def m_split_leak(dst):      # a test compound's skeleton also appears in train
    def f(d):
        d.loc[d[d.split == 'test'].index[0], 'inchikey_block'] = d[d.split == 'train'].inchikey_block.iloc[0]; return d
    edit_csv(find(dst, 'scaffold_split.csv'), f)

def m_scaffold_regress(dst):  # a scaffold group's members spread across splits (moves 20 test rows to train)
    def f(d): d.loc[d[d.split == 'test'].index[:20], 'split'] = 'train'; return d
    edit_csv(find(dst, 'scaffold_split.csv'), f)

def m_stale_doc(dst):       # README quotes a number that is not in the data
    p = os.path.join(dst, 'README.md'); t = open(p, encoding='utf-8').read()
    f = pd.read_csv(os.path.join(dst, 'results', 'classical_baselines_comparison.csv'))
    n = f"{f[f.Primary == True]['AUC-ROC'].iloc[0]:.3f}"
    assert n in t
    open(p, 'w', encoding='utf-8').write(t.replace(n, '0.999'))

def m_drop_floor(dst):      # report stops printing the AUC-PR floor
    p = os.path.join(dst, 'results', 'Phase1_Final_Report.md'); t = open(p, encoding='utf-8').read()
    f = pd.read_csv(os.path.join(dst, 'results', 'classical_baselines_comparison.csv'))
    fl = f[f.Modality == 'Baseline']['AUC-PR'].iloc[0]
    open(p, 'w', encoding='utf-8').write(t.replace(f"{fl:.3f}", "x.xxx"))
    q = os.path.join(dst, 'README.md'); u = open(q, encoding='utf-8').read()
    open(q, 'w', encoding='utf-8').write(u.replace(f"{fl:.3f}", "x.xxx"))

def m_hERG(dst):            # framing regresses to hERG prediction
    p = os.path.join(dst, 'README.md')
    open(p, 'a', encoding='utf-8').write("\nWe predict hERG blockade.\n")

# name -> (mutation, substring expected among FAILED check names)
MUT = {
    'M1': (m_expr_value, '7f'), 'M2': (m_label_flip, '2'), 'M3': (m_dup_pert, '1f'),
    'M4': (m_drop_gene, '6b'), 'M5': (m_split_leak, '4'), 'M6': (m_scaffold_regress, '5c'),
    'M7': (m_stale_doc, '10'), 'M8': (m_drop_floor, '17'), 'M9': (m_hERG, '19'),
}


def run(name):
    fn, expect = MUT[name]
    tmp, dst = make_copy()
    try:
        fn(dst)
        r = subprocess.run([sys.executable, os.path.join(dst, 'scripts', 'verify_pipeline.py')],
                           capture_output=True, text=True, cwd=dst)
        failed = [l[7:] for l in r.stdout.splitlines() if l.startswith('[FAIL]')]
        caught = r.returncode != 0 and any(x.startswith(expect) for x in failed)
        print(f"{name} {fn.__name__:18s} -> {'CAUGHT' if caught else 'MISSED'}  (expected check {expect}; failed: {[x.split()[0] for x in failed]})")
        if r.returncode != 0 and not failed:
            print(r.stderr[-500:])
        return caught
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    names = [a for a in sys.argv[1:] if a != 'all'] or list(MUT)
    if 'all' in sys.argv[1:]:
        names = list(MUT)
    ok = [run(n) for n in names]
    print(f"\n{sum(ok)}/{len(ok)} injected errors caught")
    sys.exit(0 if all(ok) else 1)
