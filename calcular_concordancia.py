"""
Calcula las métricas de concordancia entre los 5 modelos a partir de
resultados_clustering_v2.csv (word_normalized, modelo, n_ocurrencias, k_clusters,
pct_ruido, presenta_multiples_clusters).

IMPORTANTE — alcance de lo que estas métricas pueden afirmar: `k_clusters` y
`presenta_multiples_clusters` describen la ESTRUCTURA de clusters observada en
los embeddings de cada modelo, no polisemia léxica validada (ver limitación
documentada en clusterizar_sentidos.py y en el protocolo, sección 9). Por lo
tanto, el Kappa/Jaccard/Fleiss calculados aquí miden CONCORDANCIA EN
ESTRUCTURA DE EMBEDDINGS entre modelos, no concordancia en detección de
polisemia real.

Uso:
    python calcular_concordancia.py
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

RESULTADOS_CSV = "resultados_clustering_v2.csv"
MODELOS = ["mex_state", "mex_large", "beto", "bertin", "robertuito"]
N_COMPARACIONES = len(MODELOS) * (len(MODELOS) - 1) // 2  # 10
ALPHA_BONFERRONI = 0.05 / N_COMPARACIONES  # 0.005 (sección 6 del protocolo)
N_BOOTSTRAP = 1000
UMBRAL_CONSENSO = 3  # de 5 modelos, para definir "palabra de consenso" (sección 6/8)

rng = np.random.default_rng(42)


def bootstrap_kappa_ci(y1, y2, n_boot=N_BOOTSTRAP, alpha=ALPHA_BONFERRONI):
    n = len(y1)
    idx = np.arange(n)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        # kappa indefinido si todo el sample cae en una sola clase en ambos raters
        if len(set(y1[sample])) < 2 and len(set(y2[sample])) < 2:
            continue
        boots.append(cohen_kappa_score(y1[sample], y2[sample]))
    if not boots:
        return np.nan, np.nan
    lower = np.percentile(boots, 100 * alpha / 2)
    upper = np.percentile(boots, 100 * (1 - alpha / 2))
    return lower, upper


def fleiss_kappa(tabla_bool):
    """tabla_bool: DataFrame (palabra x modelo) booleano, sin NaN."""
    n_items, n_raters = tabla_bool.shape
    votos_si = tabla_bool.sum(axis=1).values.astype(float)
    votos_no = n_raters - votos_si
    P_i = (votos_si**2 + votos_no**2 - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_si = votos_si.sum() / (n_items * n_raters)
    p_no = 1 - p_si
    P_e = p_si**2 + p_no**2
    return (P_bar - P_e) / (1 - P_e)


def main():
    df = pd.read_csv(RESULTADOS_CSV)
    # presenta_multiples_clusters puede venir como True/False/NaN (None = "no determinado" o no evaluada)
    df["presenta_multiples_clusters"] = df["presenta_multiples_clusters"].map({"True": True, "False": False, True: True, False: False})

    tabla = df.pivot(index="word_normalized", columns="modelo", values="presenta_multiples_clusters")[MODELOS]

    tasas_base = tabla.mean(skipna=True)
    print("=== Tasa base de estructura multi-cluster por modelo (no es tasa de polisemia validada) ===")
    print(tasas_base.to_string(), "\n")

    tabla_completa = tabla.dropna()
    n_final = len(tabla_completa)
    print(f"Palabras evaluadas en los 5 modelos a la vez (excluye 'no determinado'/'no evaluada'): {n_final}")
    print(f"n mínimo recomendado (sección 6 del protocolo, 10 comparaciones): {int(np.ceil(1.96**2 * 0.8*0.2 / (0.05**2 * 0.5**2)))} aprox. con alpha sin corregir; con Bonferroni ~2017\n")

    # --- Cohen's Kappa pareado + Jaccard + bootstrap CI ---
    filas_kappa = []
    for m1, m2 in combinations(MODELOS, 2):
        y1 = tabla_completa[m1].astype(bool).values
        y2 = tabla_completa[m2].astype(bool).values
        kappa = cohen_kappa_score(y1, y2)
        ic_low, ic_high = bootstrap_kappa_ci(y1, y2)
        union = np.sum(y1 | y2)
        jaccard = np.sum(y1 & y2) / union if union > 0 else np.nan
        filas_kappa.append({
            "modelo_1": m1, "modelo_2": m2, "cohen_kappa": kappa,
            "ic_995_low": ic_low, "ic_995_high": ic_high, "jaccard": jaccard,
        })
    kappa_df = pd.DataFrame(filas_kappa)
    print("=== Cohen's Kappa pareado (10 combinaciones, IC 99.5% por Bonferroni) ===")
    print(kappa_df.to_string(index=False), "\n")

    # --- Fleiss' Kappa global ---
    fleiss = fleiss_kappa(tabla_completa.astype(bool))
    print(f"=== Fleiss' Kappa global (5 modelos, n={n_final}) ===\n{fleiss:.4f}\n")

    # --- Índice de consenso por modelo ---
    consenso = {m: [] for m in MODELOS}
    for _, row in kappa_df.iterrows():
        consenso[row["modelo_1"]].append(row["cohen_kappa"])
        consenso[row["modelo_2"]].append(row["cohen_kappa"])
    consenso_series = pd.Series({m: np.mean(v) for m, v in consenso.items()}).sort_values(ascending=False)
    print("=== Índice de consenso por modelo (promedio de Kappa contra los otros 4) ===")
    print(consenso_series.to_string(), "\n")

    # --- Spearman en k_clusters, sobre palabras de consenso (>=3/5 con estructura multi-cluster) ---
    tabla_k = df.pivot(index="word_normalized", columns="modelo", values="k_clusters")[MODELOS]
    es_consenso = tabla_completa.sum(axis=1) >= UMBRAL_CONSENSO
    palabras_consenso = tabla_completa[es_consenso].index
    tabla_k_consenso = tabla_k.loc[palabras_consenso].dropna()
    print(f"Palabras de consenso (>={UMBRAL_CONSENSO}/5 modelos coinciden en estructura multi-cluster), con k válido en los 5: {len(tabla_k_consenso)}")
    print("(recomendado en protocolo: >=40-50 para esta parte)\n")

    filas_spearman = []
    for m1, m2 in combinations(MODELOS, 2):
        if len(tabla_k_consenso) >= 3:
            rho, p = spearmanr(tabla_k_consenso[m1], tabla_k_consenso[m2])
        else:
            rho, p = np.nan, np.nan
        filas_spearman.append({"modelo_1": m1, "modelo_2": m2, "spearman_rho": rho, "p_value": p})
    spearman_df = pd.DataFrame(filas_spearman)
    print("=== Correlación de Spearman en número de clusters (k), palabras de consenso ===")
    print(spearman_df.to_string(index=False), "\n")

    # Guardar todo
    kappa_df.to_csv("concordancia_kappa_pareado.csv", index=False)
    consenso_series.to_csv("concordancia_indice_consenso.csv")
    spearman_df.to_csv("concordancia_spearman_k.csv", index=False)
    tasas_base.to_csv("concordancia_tasas_base.csv")
    print("Guardado: concordancia_kappa_pareado.csv, concordancia_indice_consenso.csv, "
          "concordancia_spearman_k.csv, concordancia_tasas_base.csv")


if __name__ == "__main__":
    main()