import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, ks_2samp, wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc as compute_auc,
    precision_recall_fscore_support,
    silhouette_score,
    davies_bouldin_score
)
from tensorflow.keras.models import load_model

# ------------------------
# Configurações
# ------------------------
DATA_DIR       = "../data/processed"           # onde estão as pastas FA, FP, KP, CP
MODEL_BASE_DIR = "models/Autoencoders"      # Autoencoder_<label>/autoencoder_<label>.keras
LABELS         = ["FA", "FP", "KP", "CP"]
N_POINTS       = 256
BATCH_SIZE     = 32

# ------------------------
# Carrega curvas de cada classe
# ------------------------
def load_data(label):
    path = os.path.join(DATA_DIR, label)
    X = []
    for fn in os.listdir(path):
        if fn.endswith(".npy"):
            arr = np.load(os.path.join(path, fn))
            if arr.shape[0] == N_POINTS:
                X.append(arr[..., np.newaxis])
    X = np.array(X)  # shape (N, 256, 1)
    # normalização por amostra
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True)
    return X

data = {lbl: load_data(lbl) for lbl in LABELS}

# ------------------------
# Carrega autoencoders
# ------------------------
models = {}
for lbl in LABELS:
    mp = os.path.join(MODEL_BASE_DIR, f"AE_{lbl}", f"autoencoder_{lbl}.keras")
    models[lbl] = load_model(mp)

# ------------------------
# 1) Estatísticas descritivas por (AE_train, class_test)
# ------------------------
desc_stats = []
for train_lbl in LABELS:
    ae = models[train_lbl]
    for test_lbl in LABELS:
        X = data[test_lbl]
        recon = ae.predict(X, batch_size=BATCH_SIZE, verbose=0)
        mse = np.mean((X - recon)**2, axis=(1,2))
        mu, med = mse.mean(), np.median(mse)
        sd = mse.std()
        cv = sd/mu if mu!=0 else np.nan
        sk = skew(mse)
        kt = kurtosis(mse)
        p10, p25, p75, p90 = np.percentile(mse, [10,25,75,90])
        out5 = np.sum(mse > mu+3*sd)
        desc_stats.append({
            'AE_train':  train_lbl,
            'class_test':test_lbl,
            'mean':      mu,
            'median':    med,
            'std':       sd,
            'cv':        cv,
            'skew':      sk,
            'kurtosis':  kt,
            'p10':       p10,
            'p25':       p25,
            'p75':       p75,
            'p90':       p90,
            'n_out_3sigma':  int(out5),
            'pct_out_3sigma': out5/len(mse)
        })
df_desc = pd.DataFrame(desc_stats)

# ------------------------
# 2) Métricas de separação e 3) Classificação binária
# ------------------------
sep_stats = []
bin_stats = []
for train_lbl in LABELS:
    ae = models[train_lbl]
    # agrupa todas as classes
    X_all = np.vstack([data[lbl] for lbl in LABELS])
    y_true = np.hstack([[1]*len(data[train_lbl])] + [[0]*len(data[lbl]) for lbl in LABELS if lbl!=train_lbl])
    recon_all = ae.predict(X_all, batch_size=BATCH_SIZE, verbose=0)
    mse_all = np.mean((X_all - recon_all)**2, axis=(1,2))
    # ROC AUC
    roc_auc = roc_auc_score(y_true, -mse_all)    # -mse para sinal alto = positivo
    # PR AUC
    prec, rec, th = precision_recall_curve(y_true, -mse_all)
    pr_auc = compute_auc(rec, prec)
    # KS statistic
    mse_pos = mse_all[y_true==1]
    mse_neg = mse_all[y_true==0]
    ks_stat, _ = ks_2samp(mse_pos, mse_neg)
    # EMD (Wasserstein) e JS divergence
    emd = wasserstein_distance(mse_pos, mse_neg)
    # JS requires distribuição de prob.
    hist_p, bin_edges = np.histogram(mse_pos, bins=100, density=True)
    hist_n, _         = np.histogram(mse_neg, bins=bin_edges, density=True)
    js_div = jensenshannon(hist_p+1e-10, hist_n+1e-10)  # evitar zeros
    sep_stats.append({
        'AE_train': train_lbl,
        'roc_auc':  roc_auc,
        'pr_auc':   pr_auc,
        'ks_stat':  ks_stat,
        'emd':      emd,
        'js_div':   js_div
    })
    # threshold = ponto médio entre medianas (pos vs neg)
    thresh = (np.median(mse_pos) + np.median(mse_neg))/2
    y_pred = (mse_all <= thresh).astype(int)
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, average='binary')
    bin_stats.append({
        'AE_train':    train_lbl,
        'threshold':   thresh,
        'precision':   p,
        'recall':      r,
        'f1_score':    f1,
        'support_pos': sup
    })

df_sep = pd.DataFrame(sep_stats)
df_bin = pd.DataFrame(bin_stats)

# ------------------------
# 4) Métricas de clustering 1D (silhouette, Davies-Bouldin)
# ------------------------
cluster_stats = []
for train_lbl in LABELS:
    ae = models[train_lbl]
    X_all = np.vstack([data[lbl] for lbl in LABELS])
    y_lbl = np.hstack([[train_lbl]*len(data[train_lbl])] +
                      [[f"not_{train_lbl}"]*len(data[lbl]) for lbl in LABELS if lbl!=train_lbl])
    recon_all = ae.predict(X_all, batch_size=BATCH_SIZE, verbose=0)
    mse_all = np.mean((X_all - recon_all)**2, axis=(1,2)).reshape(-1,1)
    # silhouette requires at least 2 labels
    try:
        sil = silhouette_score(mse_all, y_lbl, metric='euclidean')
    except:
        sil = np.nan
    try:
        db = davies_bouldin_score(mse_all, y_lbl)
    except:
        db = np.nan
    cluster_stats.append({
        'AE_train': train_lbl,
        'silhouette':          sil,
        'davies_bouldin':      db
    })
df_cluster = pd.DataFrame(cluster_stats)

# ------------------------
# Output
# ------------------------
print("\n=== Estatísticas Descritivas ===")
df_desc.pivot(index="AE_train", columns="class_test", values="mean")


print("\n=== Métricas de Separação (ROC AUC, PR AUC, KS, EMD, JS) ===")
print(df_sep.set_index("AE_train").round(4))

print("\n=== Métricas Binárias (threshold, precision, recall, f1) ===")
print(df_bin.set_index("A0,,0E_train").round(4))

print("\n=== Métricas de Clustering 1D ===")
print(df_cluster.set_index("AE_train").round(4))

# Salvar tudo em CSV
df_desc.to_csv("ae_data_quality_desc_stats.csv", index=False)
df_sep.to_csv("ae_data_quality_separation.csv", index=False)
df_bin.to_csv("ae_data_quality_binary.csv", index=False)
df_cluster.to_csv("ae_data_quality_clustering.csv", index=False)
