import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

def compute_resemblance(df1: pd.DataFrame, df2: pd.DataFrame, columns=None, name1="Dataset1", name2="Dataset2") -> dict:
    if columns is None:
        columns = df1.columns.intersection(df2.columns)

    results = {}
    js_scores, ks_scores, w_scores = [], [], []

    for col in columns:
        if col.lower() == "gender":
            continue
        if not np.issubdtype(df1[col].dtype, np.number) or not np.issubdtype(df2[col].dtype, np.number):
            continue

        hist1, _ = np.histogram(df1[col].dropna(), bins=20, density=True)
        hist2, _ = np.histogram(df2[col].dropna(), bins=20, density=True)
        js = jensenshannon(hist1, hist2)

        ks_stat, ks_pvalue = ks_2samp(df1[col].dropna(), df2[col].dropna())
        w = wasserstein_distance(df1[col].dropna(), df2[col].dropna())

        results[col] = {
            "JS Divergence": js,
            "KS D-Statistic": ks_stat,
            "KS p-value": ks_pvalue,
            "Wasserstein": w
        }

        js_scores.append(js)
        ks_scores.append(ks_stat)
        w_scores.append(w)

    overall = {
        "JS Divergence": np.mean(js_scores) if js_scores else None,
        "KS D-Statistic": np.mean(ks_scores) if ks_scores else None,
        "Wasserstein": np.mean(w_scores) if w_scores else None
    }

    return {
        "results": results,
        "overall": overall,
        "datasets": (name1, name2)
    }
