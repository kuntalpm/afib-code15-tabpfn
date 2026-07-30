"""
Stage 9 - External replication in PTB-XL (Methods 2.11, Results 3.11).

PTB-XL cannot reproduce the temporal design (only 57 of 1,225 AF/AFL
patients have a sinus record preceding the first AF record, median
interval 10 days); it is used only to test whether the cross-sectional
feature-outcome relationships reproduce in an independent population, on
different hardware (500 Hz vs 400 Hz), in a different country and era.

Two versions of the V1 feature are reported, in the order they were
actually run:
  1. proxy (identical extraction code to the pre-final CODE-15% pipeline,
     rescaled to 500 Hz) -- this is what "direct transfer"/"retrain" first
     used, giving AUROC 0.584 for both.
  2. true peak P-wave amplitude, re-extracted directly from PTB-XL
     waveforms with the same delineation pipeline as CODE-15% (99.7%
     coverage, 673/675) -- direct transfer 0.596, retrained 0.620, the
     latter's CI overlapping the CODE-15% seed-averaged estimate.

Cohort: 135 AF/AFL cases with a sinus record at any prior age (not
necessarily immediately prior), matched 1:4 to controls on age and sex.

Input : ptbxl_features.csv (proxy features + RMSSD + PR_med),
        ptbxl_amplitude.csv (re-extracted true V1 amplitude),
        cohort_clean.csv (CODE-15% derivation cohort, for the transfer model)
Output: printed AUROC for all four combinations
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            boots.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(boots, [2.5, 97.5])


def retrain_cv(m_frame, features, n_repeats=10):
    y = (m_frame["split"] == "vaka").astype(int).values
    X = SimpleImputer().fit_transform(m_frame[features].values)
    P = np.zeros((n_repeats, len(y)))
    for t in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=t)
        for tr, te in skf.split(X, y, groups=m_frame.patient_id.values):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(X[tr], y[tr])
            P[t, te] = model.predict_proba(X[te])[:, 1]
    return y, P.mean(0)


def main():
    derivation = pd.read_csv("cohort_clean.csv")
    derivation["Pamp_V1"] = derivation.get("Pamp_6")
    F3 = ["RMSSD", "PR_med", "Pamp_V1"]
    y_dev = (derivation["split"] == "vaka").astype(int).values
    X_dev = SimpleImputer().fit_transform(derivation[F3].values)
    derivation_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    derivation_model.fit(X_dev, y_dev)

    ptbxl = pd.read_csv("ptbxl_features.csv")  # RMSSD, PR_med, PTFV1 (proxy)
    amp = pd.read_csv("ptbxl_amplitude.csv")   # exam_id, Pamp_V1 (true)
    ptbxl_true = ptbxl.merge(amp, on="exam_id", how="inner")
    print(f"True-amplitude coverage: {len(ptbxl_true)}/{len(ptbxl)}")

    for label, frame, feat in [
        ("PROXY", ptbxl, ["RMSSD", "PR_med", "PTFV1"]),
        ("TRUE AMPLITUDE", ptbxl_true, ["RMSSD", "PR_med", "Pamp_V1"]),
    ]:
        print(f"\n=== {label} ===")
        y = (frame["split"] == "vaka").astype(int).values
        X = SimpleImputer().fit_transform(frame[feat].values)

        # direct transfer: score with derivation-fitted model (feature names
        # must align positionally with F3 = [RMSSD, PR_med, Pamp_V1])
        p_transfer = derivation_model.predict_proba(X)[:, 1]
        auc_t, ci_t = bootstrap_ci(y, p_transfer)
        print(f"  Direct transfer : {auc_t:.3f}  95% CI [{ci_t[0]:.3f},{ci_t[1]:.3f}]")

        y2, p_retrain = retrain_cv(frame, feat)
        auc_r, ci_r = bootstrap_ci(y2, p_retrain)
        print(f"  Retrained in PTB-XL: {auc_r:.3f}  95% CI [{ci_r[0]:.3f},{ci_r[1]:.3f}]")


if __name__ == "__main__":
    main()
