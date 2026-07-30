"""
Stage 7 - Age/sex matching, averaged over 11 independent realisations
(Methods 2.2, 2.9; Results 3.3-3.7).

Matching: each case is paired without replacement to one control of the
same sex and within two years of age (widened to three years if no
candidate is available). Seed 42 is the realisation fixed before any
results were seen when the cohort was first constructed and is retained
for single-realisation tables/figures; seeds 0-9 are the ten additional
realisations used to quantify matching-seed variability (Discussion §4).

This script reports, for each of 11 seeds:
  - Age+sex baseline AUROC (should be ~0.50 by construction)
  - RR-block-only AUROC
  - Atrial-block-only AUROC (PR interval + P-wave amplitude in V1)
  - Combined 3-feature AUROC (RMSSD, PR_med, Pamp_V1) -- the final model
  - H2: incremental contribution of the atrial block over RR alone

The manuscript's seed-averaged headline figures (Results 3.3) are the mean
across these 11 runs; seed 42 alone is what is shown in Table 2, Table 5
and Figures 3, 4, 6.

Input : rr_features.csv, pwave_features.csv, ptfv1_true.csv,
        cohort_raw.csv, exams.csv (for is_male)
Output: cohort_clean.csv (seed-42 matched cohort, used by downstream
        scripts), printed per-seed and averaged AUROC table
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

RR = ["meanRR", "RMSSD"]
ATRIAL = ["PR_med", "Pamp_V1"]
FINAL3 = ["RMSSD", "PR_med", "Pamp_V1"]
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def build_dataset():
    r = pd.read_csv("rr_features.csv")
    p = pd.read_csv("pwave_features.csv")
    t = pd.read_csv("ptfv1_true.csv")
    c = pd.read_csv("cohort_raw.csv")
    e = pd.read_csv("data/exams.csv")[["exam_id", "is_male"]]

    d = (
        r.merge(p, on="exam_id")
        .merge(t, on="exam_id", how="left")
        .merge(c, on="exam_id")
        .merge(e, on="exam_id")
    )
    d = d[d["split"].isin(["vaka", "kontrol"])]
    d = d[d.n_R >= 6].reset_index(drop=True)
    d["Pamp_V1"] = d.get("Pamp_6")
    return d


def match_once(d, seed):
    cases = d[d["split"] == "vaka"]
    controls = d[d["split"] == "kontrol"]
    rng = np.random.default_rng(seed)
    used, pairs = set(), []
    for _, case in cases.sort_values("age_at_index", ascending=False).iterrows():
        pool = controls[
            (~controls.exam_id.isin(used))
            & (controls.is_male == case.is_male)
            & (abs(controls.age_at_index - case.age_at_index) <= 2)
        ]
        if pool.empty:
            pool = controls[
                (~controls.exam_id.isin(used))
                & (abs(controls.age_at_index - case.age_at_index) <= 3)
            ]
        if pool.empty:
            continue
        chosen = pool.iloc[int(rng.integers(len(pool)))]
        used.add(chosen.exam_id)
        pairs.append((case.exam_id, chosen.exam_id))
    matched_ids = [a for a, _ in pairs] + [b for _, b in pairs]
    return d[d.exam_id.isin(matched_ids)].reset_index(drop=True)


def cv_auc(matched, features, n_repeats=10):
    y = (matched["split"] == "vaka").astype(int).values
    X = SimpleImputer().fit_transform(matched[features].values)
    P = np.zeros((n_repeats, len(y)))
    for t in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=t)
        for tr, te in skf.split(X, y, groups=matched.patient_id.values):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr], y[tr])
            P[t, te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, P.mean(0))


def main():
    d = build_dataset()
    results = {"M0": [], "RR": [], "Atrial": [], "Final3": [], "H2": []}

    for seed in SEEDS:
        m = match_once(d, seed)
        if seed == 42:
            m.to_csv("cohort_clean.csv", index=False)  # for downstream scripts

        auc_m0 = cv_auc(m, ["age_at_index", "is_male"])
        auc_rr = cv_auc(m, RR)
        auc_atrial = cv_auc(m, ATRIAL)
        auc_final3 = cv_auc(m, FINAL3)

        results["M0"].append(auc_m0)
        results["RR"].append(auc_rr)
        results["Atrial"].append(auc_atrial)
        results["Final3"].append(auc_final3)
        results["H2"].append(auc_final3 - auc_rr)

        print(f"seed {seed:3d}: M0={auc_m0:.3f} RR={auc_rr:.3f} "
              f"Atrial={auc_atrial:.3f} Final3={auc_final3:.3f} "
              f"H2={auc_final3 - auc_rr:+.3f}")

    print("\n=== 11-seed averages ===")
    for k, v in results.items():
        v = np.array(v)
        print(f"  {k:8s} mean={v.mean():.3f} sd={v.std():.4f} "
              f"range=[{v.min():.3f},{v.max():.3f}]")
        if k == "H2":
            print(f"           positive in {int((v > 0).sum())}/{len(v)} seeds")


if __name__ == "__main__":
    main()
