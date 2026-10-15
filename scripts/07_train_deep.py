"""
07_train_deep.py -- Phase 2: two INDEPENDENT deep models on the same compounds and the same splits as the baselines.

    gin          structure : 3-layer Graph Isomorphism Network over the RDKit molecular graph (mean readout, jumping-knowledge
                             concat of layer readouts), edge types are not used (plain GIN).
    transformer  expression: 2-layer, 4-head Transformer encoder over the 978 L1000 landmark genes. Each gene is one token =
                             learned gene-identity embedding + a learned projection of that gene's z-score. No positional
                             encoding (gene order is arbitrary). Mean-pooled tokens -> MLP head.

Protocol (identical for both models; nothing is tuned on the test set):
  * class-weighted BCE (pos_weight = n_neg / n_pos of the TRAINING set, scripts.utils.compute_class_weight)
  * AdamW, linear warm-up then cosine decay; model selected at the epoch of best validation AUC-ROC (early stopping)
  * decision threshold = the validation threshold maximising balanced accuracy (used only for balanced accuracy)
  * expression is standardised with a scaler fit on the training compounds only

  frozen : train on scaffold_split train, select on val, score the frozen test set, SEEDS independent training seeds.
           Reported: per-seed AUC-ROC, and the seed-ensemble (mean probability) with a 95% test-set bootstrap CI.
  cv     : the SAME scaffold-grouped 5-fold folds as 05_split_stability.py (repeat REPEAT); inside every training fold a
           scaffold-grouped ~1/6 is held out for early stopping. Out-of-fold seed-ensemble predictions are pooled over all
           compounds (AUC-ROC + bootstrap CI), directly comparable with the pooled-CV numbers of the baselines.
  report : writes results/deep_models_report.md (computed comparison with the RF / LR baselines, paired bootstrap).

Usage:
    python scripts/07_train_deep.py frozen gin|transformer [--seeds 5] [--max-epochs 150]
    python scripts/07_train_deep.py cv     gin|transformer [--seeds 3] [--repeat 0]
    python scripts/07_train_deep.py report
"""
import os, sys, time, argparse, copy, math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from scripts.utils import compute_class_weight, scaffold_group_folds, set_seed

SEED, N_BOOT = CFG.RANDOM_SEED, 2000
R = CFG.RESULTS_DIR

# --------------------------------------------------------------------------- data
scaf = pd.read_csv(os.path.join(CFG.DATA_DIR, 'splits', 'scaffold_split.csv'))
expr = pd.read_csv(CFG.EXPRESSION_CSV, index_col=0)
assert set(expr.index) == set(scaf.parent_smiles), "expression / split compound sets differ"
scaf = scaf.reset_index(drop=True)
SMILES = scaf.parent_smiles.tolist()
Y = scaf.cardiotox_label.values.astype(np.float32)
GROUPS = scaf.scaffold.values.astype(str)
SPLIT = scaf.split.values
X_EXPR = expr.loc[SMILES].values.astype(np.float32)
N = len(scaf)


def atom_features(atom):
    sym = atom.GetSymbol()
    onehot = [1.0 if sym == t else 0.0 for t in CFG.ATOM_TYPES[:-1]]
    onehot.append(1.0 if sym not in CFG.ATOM_TYPES[:-1] else 0.0)
    return onehot + [atom.GetTotalDegree() / 6.0, float(atom.GetIsAromatic()), float(atom.GetFormalCharge()),
                     atom.GetTotalNumHs() / 4.0]


def build_graph(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    return x, np.array([src, dst], dtype=np.int64).reshape(2, -1)


GRAPHS = None   # built lazily (only the GIN needs RDKit graphs)


def get_graphs():
    global GRAPHS
    if GRAPHS is None:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        GRAPHS = [build_graph(s) for s in SMILES]
        assert GRAPHS[0][0].shape[1] == CFG.NODE_FEATURE_DIM, (GRAPHS[0][0].shape, CFG.NODE_FEATURE_DIM)
    return GRAPHS


# --------------------------------------------------------------------------- models
import torch
import torch.nn as nn
torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)))


def make_head(in_dim, dims=None, dropout=None):
    dims = dims or CFG.MLP_HIDDEN_DIMS
    dropout = CFG.MLP_DROPOUT if dropout is None else dropout
    layers, d = [], in_dim
    for h in dims:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class GIN(nn.Module):
    def __init__(self):
        super().__init__()
        H, L = CFG.GNN_HIDDEN_DIM, CFG.GNN_NUM_LAYERS
        self.inp = nn.Linear(CFG.NODE_FEATURE_DIM, H)
        self.eps = nn.Parameter(torch.zeros(L))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(H, H), nn.BatchNorm1d(H), nn.ReLU(), nn.Linear(H, H)) for _ in range(L)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(H) for _ in range(L)])
        self.drop = nn.Dropout(CFG.GNN_DROPOUT)
        self.proj = nn.Linear(H * L, CFG.EMBED_DIM)
        self.head = make_head(CFG.EMBED_DIM)

    def forward(self, x, edge_index, batch, n_graphs):
        h = self.inp(x)
        outs = []
        for l, (mlp, bn) in enumerate(zip(self.mlps, self.bns)):
            agg = torch.zeros_like(h).index_add_(0, edge_index[1], h[edge_index[0]])
            h = mlp((1 + self.eps[l]) * h + agg)
            h = self.drop(torch.relu(bn(h)))
            pooled = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
            counts = torch.zeros(n_graphs).index_add_(0, batch, torch.ones(len(batch))).clamp(min=1).unsqueeze(1)
            outs.append(pooled / counts)
        return self.head(torch.relu(self.proj(torch.cat(outs, 1)))).squeeze(-1)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block using fused scaled-dot-product attention (no attention dropout)."""
    def __init__(self, d, nhead, ff, dropout):
        super().__init__()
        self.nhead = nhead
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.nhead, D // self.nhead).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.out(a))
        return x + self.drop(self.ff(self.n2(x)))


TRANS_D = CFG.TRANS_D_MODEL      # overridable with --d-model (compute-reduced variant); feed-forward width = 4 * d


class ExprTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = TRANS_D
        self.gene_emb = nn.Embedding(CFG.NUM_GENES, d)           # learned gene identity (no positional encoding)
        self.val_proj = nn.Linear(1, d)                          # projection of the gene's z-score
        self.blocks = nn.ModuleList([EncoderBlock(d, CFG.TRANS_NHEAD, 4 * d, CFG.TRANS_DROPOUT) for _ in range(CFG.TRANS_NUM_LAYERS)])
        self.norm = nn.LayerNorm(d)
        self.head = make_head(d)

    def forward(self, x):
        h = self.gene_emb.weight.unsqueeze(0) + self.val_proj(x.unsqueeze(-1))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h).mean(1)).squeeze(-1)


# --------------------------------------------------------------------------- batching
def graph_batch(idx):
    g = get_graphs()
    xs, eis, bs, off = [], [], [], 0
    for k, i in enumerate(idx):
        x, ei = g[i]
        xs.append(x); eis.append(ei + off); bs.append(np.full(len(x), k, dtype=np.int64)); off += len(x)
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(eis, 1)),
            torch.from_numpy(np.concatenate(bs)), len(idx))


def forward_idx(model, kind, idx, scaler_mu=None, scaler_sd=None):
    if kind == 'gin':
        return model(*graph_batch(idx))
    x = (X_EXPR[idx] - scaler_mu) / scaler_sd
    return model(torch.from_numpy(x.astype(np.float32)))


def predict(model, kind, idx, mu, sd, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            out.append(torch.sigmoid(forward_idx(model, kind, idx[s:s + bs], mu, sd)).numpy())
    return np.concatenate(out)


def best_threshold(y, p):
    """Threshold maximising balanced accuracy on the validation set."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))]) if np.isfinite(thr[int(np.argmax(j))]) else 0.5


def train_one(kind, tr, va, targets, seed, max_epochs, patience):
    """Train one model; return {name: probabilities for each array in targets}, val AUC, best epoch, threshold."""
    assert len(set(SMILES[i] for i in tr) & set(SMILES[i] for i in va)) == 0
    assert len(set(GROUPS[tr]) & set(GROUPS[va])) == 0, "scaffold shared between train and validation"
    for t in targets.values():
        assert len(set(GROUPS[tr]) & set(GROUPS[t])) == 0, "scaffold shared between train and evaluation set"
    set_seed(seed)
    model = GIN() if kind == 'gin' else ExprTransformer()
    mu = X_EXPR[tr].mean(0); sd = X_EXPR[tr].std(0) + 1e-6
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    pos_w = torch.tensor(compute_class_weight(Y[tr]), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    warm = CFG.LR_WARMUP_EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, max_epochs - warm))))
    rng = np.random.RandomState(seed)
    best = (-1.0, 1e9, None, 0)
    since = 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), CFG.BATCH_SIZE):
            b = perm[s:s + CFG.BATCH_SIZE]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = lossf(forward_idx(model, kind, b, mu, sd), torch.from_numpy(Y[b]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        pv = predict(model, kind, va, mu, sd)
        auc = roc_auc_score(Y[va], pv)
        vloss = float(lossf(torch.logit(torch.from_numpy(pv).clamp(1e-6, 1 - 1e-6)), torch.from_numpy(Y[va])))
        if auc > best[0] + 1e-9 or (abs(auc - best[0]) <= 1e-9 and vloss < best[1]):
            best = (auc, vloss, copy.deepcopy(model.state_dict()), ep)
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best[2])
    thr = best_threshold(Y[va], predict(model, kind, va, mu, sd))
    return {k: predict(model, kind, t, mu, sd) for k, t in targets.items()}, best[0], best[3], thr


def boot_idx(n, seed=SEED, n_boot=N_BOOT):
    rs = np.random.RandomState(seed)
    return [rs.randint(0, n, n) for _ in range(n_boot)]


def boot_ci(y, s, idxs):
    vals = [roc_auc_score(y[i], s[i]) for i in idxs if y[i].min() != y[i].max()]
    return tuple(np.percentile(vals, [2.5, 97.5]))


# --------------------------------------------------------------------------- modes
def run_frozen(kind, seeds, max_epochs, patience):
    tr, va, te = [np.where(SPLIT == s)[0] for s in ('train', 'val', 'test')]
    rows, preds, t0 = [], [], time.time()
    for s in range(seeds):
        out, vauc, ep, thr = train_one(kind, tr, va, {'test': te}, SEED + s, max_epochs, patience)
        p = out['test']; preds.append(p)
        rows.append(dict(model=kind, seed=SEED + s, val_auc=vauc, best_epoch=ep, test_auc=roc_auc_score(Y[te], p),
                         test_bal_acc=balanced_accuracy_score(Y[te], (p >= thr).astype(int)), threshold=thr))
        print(f"[{kind}] seed {SEED+s}: val AUC {vauc:.3f} (epoch {ep})  test AUC {rows[-1]['test_auc']:.3f}  "
              f"bal-acc {rows[-1]['test_bal_acc']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, f'deep_frozen_{kind}_seeds.csv'), index=False)
    pd.DataFrame({'parent_smiles': [SMILES[i] for i in te], 'y': Y[te], **{f'seed{k}': p for k, p in enumerate(preds)},
                  'ensemble': np.mean(preds, 0)}).to_csv(os.path.join(R, f'deep_frozen_{kind}_test_predictions.csv'), index=False)


def run_cv(kind, seeds, repeat, max_epochs, patience):
    fold = scaffold_group_folds(GROUPS, 5, 5000 + repeat)
    oof = np.zeros(N); t0 = time.time()
    rows = []
    for f in range(5):
        te = np.where(fold == f)[0]; pool = np.where(fold != f)[0]
        inner = scaffold_group_folds(GROUPS[pool], 6, 7000 + 10 * repeat + f)
        va, tr = pool[inner == 0], pool[inner != 0]
        ps = []
        for s in range(seeds):
            out, vauc, ep, _ = train_one(kind, tr, va, {'test': te}, SEED + s, max_epochs, patience)
            ps.append(out['test'])
            rows.append(dict(model=kind, repeat=repeat, fold=f, seed=SEED + s, val_auc=vauc, best_epoch=ep,
                             fold_auc=roc_auc_score(Y[te], out['test'])))
            print(f"[{kind}] cv fold {f} seed {SEED+s}: val {vauc:.3f} ep {ep}  fold AUC {rows[-1]['fold_auc']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
        oof[te] = np.mean(ps, 0)
    pd.DataFrame(rows).to_csv(os.path.join(R, f'deep_cv_{kind}_folds.csv'), index=False)
    pd.DataFrame({'parent_smiles': SMILES, 'fold': fold, 'y': Y, 'oof': oof}).to_csv(os.path.join(R, f'deep_cv_{kind}_oof.csv'), index=False)
    print(f"[{kind}] pooled CV AUC-ROC {roc_auc_score(Y, oof):.3f}")


def paired_diff(y, a, b, idxs):
    vals = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idxs if y[i].min() != y[i].max()]
    return roc_auc_score(y, a) - roc_auc_score(y, b), tuple(np.percentile(vals, [2.5, 97.5]))


def run_report():
    te = np.where(SPLIT == 'test')[0]
    # baseline test scores come from 04_baselines.py (never refitted here: random forests differ across library versions)
    bp = pd.read_csv(os.path.join(R, 'baseline_test_predictions.csv'))
    assert bp.parent_smiles.tolist() == [SMILES[i] for i in te] and (bp.y.values == Y[te]).all(), "baseline predictions are not for the frozen test set"
    p_rf, p_lr = bp['Structure: Random Forest (200 trees)'].values, bp['Biology: Logistic Regression (L2)'].values
    comp = pd.read_csv(os.path.join(R, 'classical_baselines_comparison.csv'))
    prim = comp[comp.Primary == True].set_index('Modality')
    for name, p, mod in (('RF', p_rf, 'Structure'), ('LR', p_lr, 'Biology')):
        assert abs(roc_auc_score(Y[te], p) - prim.loc[mod, 'AUC-ROC']) < 1e-9, f"{name} predictions disagree with classical_baselines_comparison.csv"
    oofb = pd.read_csv(os.path.join(R, 'split_stability_oof.csv')).set_index('parent_smiles').loc[SMILES]
    cvb = {'gin': oofb['Structure: Morgan-FP RF'].values, 'transformer': oofb['Biology: L1000 LR (L2)'].values}
    fb = {'gin': p_rf, 'transformer': p_lr}
    bname = {'gin': 'Structure RF (baseline)', 'transformer': 'Biology LR (baseline)'}
    mname = {'gin': 'GIN (structure)', 'transformer': 'Transformer (expression)'}
    idx_te, idx_all = boot_idx(len(te)), boot_idx(N)
    yte = Y[te]
    L = ["# Phase 2: Deep Models vs Baselines", "", "*Auto-generated by `scripts/07_train_deep.py report`.*", "",
         f"Cohort {N} compounds ({int(Y.sum())} toxic / {int((Y == 0).sum())} safe). Frozen scaffold test set: {len(te)} compounds, "
         f"{int((yte == 0).sum())} safe. All models use the same compounds, the same scaffold split and the same scaffold-grouped CV folds. "
         "Intervals are percentile bootstraps over compounds with fixed predictions; they do not include training variability, which is "
         "reported separately as the SD over training seeds.", ""]
    L += ["## Frozen scaffold split (test AUC-ROC)", "",
          "| Model | Seeds | Per-seed AUC-ROC (mean ± SD) | Seed-ensemble AUC-ROC | 95% CI | Balanced acc (val-chosen threshold, mean) |", "|---|:-:|:-:|:-:|:-:|:-:|"]
    for base, p, nm in (('rf', p_rf, bname['gin']), ('lr', p_lr, bname['transformer'])):
        a = roc_auc_score(yte, p); lo, hi = boot_ci(yte, p, idx_te)
        L.append(f"| {nm} | – | – | {a:.3f} | [{lo:.3f}, {hi:.3f}] | {balanced_accuracy_score(yte, (p >= .5).astype(int)):.3f} |")
    verdict = []
    for kind in ('gin', 'transformer'):
        f_seeds = os.path.join(R, f'deep_frozen_{kind}_seeds.csv')
        if not os.path.exists(f_seeds):
            continue
        sd = pd.read_csv(f_seeds); pr = pd.read_csv(os.path.join(R, f'deep_frozen_{kind}_test_predictions.csv'))
        assert pr.parent_smiles.tolist() == [SMILES[i] for i in te], "test predictions are not for the frozen test set"
        ens = pr.ensemble.values; a = roc_auc_score(yte, ens); lo, hi = boot_ci(yte, ens, idx_te)
        L.append(f"| {mname[kind]} | {len(sd)} | {sd.test_auc.mean():.3f} ± {sd.test_auc.std():.3f} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | {sd.test_bal_acc.mean():.3f} |")
        d, (dlo, dhi) = paired_diff(yte, ens, fb[kind], idx_te)
        verdict.append((kind, 'frozen', d, dlo, dhi))
    L += ["", "## Scaffold-grouped 5-fold CV, repeat 0 (pooled out-of-fold AUC-ROC over all compounds)", "",
          "| Model | Pooled AUC-ROC | 95% CI | Paired difference vs same-modality baseline | 95% CI of difference |", "|---|:-:|:-:|:-:|:-:|"]
    for kind in ('gin', 'transformer'):
        a = roc_auc_score(Y, cvb[kind]); lo, hi = boot_ci(Y, cvb[kind], idx_all)
        L.append(f"| {bname[kind]} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | – | – |")
        f_oof = os.path.join(R, f'deep_cv_{kind}_oof.csv')
        if os.path.exists(f_oof):
            o = pd.read_csv(f_oof)
            assert o.parent_smiles.tolist() == SMILES and (o.fold.values == oofb.fold.values).all(), "deep CV folds differ from baseline CV folds"
            a2 = roc_auc_score(Y, o.oof.values); lo2, hi2 = boot_ci(Y, o.oof.values, idx_all)
            d, (dlo, dhi) = paired_diff(Y, o.oof.values, cvb[kind], idx_all)
            L.append(f"| {mname[kind]} | {a2:.3f} | [{lo2:.3f}, {hi2:.3f}] | {d:+.3f} | [{dlo:+.3f}, {dhi:+.3f}] |")
            verdict.append((kind, 'cv', d, dlo, dhi))
    L += ["", "## Reading (computed)", ""]
    if not verdict:
        L.append("- No deep-model results found yet.")
    for kind, where, d, lo, hi in verdict:
        if lo > 0:
            txt = "better than the baseline (paired interval above 0)"
        elif hi < 0:
            txt = "worse than the baseline (paired interval below 0)"
        else:
            txt = "not distinguishable from the baseline (paired interval includes 0)"
        L.append(f"- **{mname[kind]}, {'frozen split' if where == 'frozen' else 'pooled CV'}:** AUC-ROC difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {txt}.")
    L += ["- A model counts as an improvement only if the paired CV interval is above 0 **and** the gain exceeds the split-to-split SD in `results/split_stability_report.md`.",
          "- Absolute signal check: a pooled-CV interval that includes 0.5 means the model has not been shown to carry any signal."]
    open(os.path.join(R, 'deep_models_report.md'), 'w', encoding='utf-8').write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['frozen', 'cv', 'report'])
    ap.add_argument('kind', nargs='?', choices=['gin', 'transformer'])
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--max-epochs', type=int, default=CFG.MAX_EPOCHS)
    ap.add_argument('--patience', type=int, default=CFG.EARLY_STOP_PATIENCE)
    a = ap.parse_args()
    if a.mode == 'report':
        run_report()
    else:
        assert a.kind, "specify gin or transformer"
        if a.mode == 'frozen':
            run_frozen(a.kind, a.seeds or 5, a.max_epochs, a.patience)
        else:
            run_cv(a.kind, a.seeds or 3, a.repeat, a.max_epochs, a.patience)
