import pandas as pd

df = pd.read_csv('results/classical_baselines_comparison.csv')
key_models = [
    'Random Classifier (Prevalence)',
    'Random Forest (200 trees)',
    'Sparse Logistic Regression (L1)',
    'Logistic Regression (L2)'
]
for m in key_models:
    row = df[df['Model'] == m]
    if len(row):
        r = row.iloc[0]
        mod = r['Modality']
        roc = r['AUC-ROC']
        roc_lo = r['AUC-ROC_95CI_Low']
        roc_hi = r['AUC-ROC_95CI_High']
        bacc = r['Balanced_Acc']
        ap = r['AUC-PR']
        ap_lo = r['AUC-PR_95CI_Low']
        ap_hi = r['AUC-PR_95CI_High']
        print(mod, '|', m)
        print('  AUC-ROC=%.4f, CI=[%.3f,%.3f], Bal-Acc=%.4f, AUC-PR=%.4f, CI=[%.3f,%.3f]' % (roc,roc_lo,roc_hi,bacc,ap,ap_lo,ap_hi))
