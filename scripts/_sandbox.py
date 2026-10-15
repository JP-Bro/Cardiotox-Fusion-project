"""Shared helper: make a throw-away copy of the repo (raw LINCS data symlinked, not copied)."""
import os, shutil, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_copy():
    tmp = tempfile.mkdtemp(prefix='cardiotox_copy_')
    dst = os.path.join(tmp, 'repo')
    ignore = shutil.ignore_patterns('.git', '__pycache__', 'legacy_week1', 'graphs', 'raw', '*.pt')
    shutil.copytree(ROOT, dst, ignore=ignore)
    os.makedirs(os.path.join(dst, 'data'), exist_ok=True)
    os.symlink(os.path.join(ROOT, 'data', 'raw'), os.path.join(dst, 'data', 'raw'))
    return tmp, dst
