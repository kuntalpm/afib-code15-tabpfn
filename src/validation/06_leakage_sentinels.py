"""
Stage 6 - Leakage sentinels (Methods 2.7).

Sentinel-1 (positive control): AF-labelled vs sinus-labelled records
distinguished using RR features alone. Confirms the extraction pipeline
functions (expected: clearly above chance; the pre-specified threshold of
0.95 assumed a longer recording than the ~8 RR intervals available here and
is not attained -- see manuscript for discussion).

Sentinel-2 (negative control): within the matched case/control cohort,
cases vs controls distinguished using RR features alone. An unexpectedly
high AUROC here would suggest AF contamination of nominally-sinus index
records rather than a genuine RR-based signal. Adjudicated using P-wave
presence (independent of RR irregularity): if the RR-based classifier's
top-scoring "case" records show P-wave delineation failure at a rate
comparable to the contamination hypothesis (~20%) rather than the observed
background rate (2-3%), contamination is indicated; otherwise not.

Input : rr_features.csv, pwave_features.csv, cohort_clean.csv (matched cohort)
Output: printed AUROC values
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

RR_FEATURES = ["meanRR", "SDNN", "RMSSD", "n_RR", "dur_s"]


def cv_auc(X, y, groups, n_splits=5, n_repeats=10, seed0=0):
    P = np.zeros((n_repeats, len(y)))
    for t in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + t)
        for tr, te in skf.split(X, y, groups=groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr], y[tr])
            P[t, te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, P.mean(0)), P.mean(0)


def main():
    r = pd.read_csv("rr_features.csv")
    p = pd.read_csv("pwave_features.csv")
    cohort = pd.read_csv("cohort_clean.csv")  # produced by 07_seed_averaged_matching.py
    d = r.merge(p, on="exam_id").merge(
        cohort[["exam_id", "patient_id", "split"]], on="exam_id"
    )
    d = d[d.n_R >= 6].reset_index(drop=True)

    print("=== Sentinel-1: AF-rhythm vs sinus, RR features only ===")
    s1 = d[d["split"].isin(["sentinel_af", "vaka", "kontrol"])]
    y1 = (s1["split"] == "sentinel_af").astype(int).values
    X1 = SimpleImputer().fit_transform(s1[RR_FEATURES].values)
    auc1, _ = cv_auc(X1, y1, s1.patient_id.values)
    print(f"  AUROC = {auc1:.3f}")

    print("\n=== Sentinel-2: case vs control (matched cohort), RR features only ===")
    s2 = d[d["split"].isin(["vaka", "kontrol"])]
    y2 = (s2["split"] == "vaka").astype(int).values
    X2 = SimpleImputer().fit_transform(s2[RR_FEATURES].values)
    auc2, probs2 = cv_auc(X2, y2, s2.patient_id.values)
    print(f"  AUROC = {auc2:.3f}")

    threshold = np.quantile(probs2[y2 == 0], 0.95)
    flagged = probs2[y2 == 1] > threshold
    print(f"\n  Cases flagged above 95th control percentile: {100*flagged.mean():.1f}%")

    n_lead_ok = s2[y2 == 1].n_lead_ok
    print(f"  Of those, P-wave delineation failure rate (<3 leads): "
          f"{100*(n_lead_ok < 3).mean():.1f}% "
          f"(contamination hypothesis predicts ~20%)")


if __name__ == "__main__":
    main()
