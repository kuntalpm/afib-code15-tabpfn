"""
Stage 11 - Formal age-by-feature interaction test (Discussion, added in
response to peer review).

Tests whether the apparent under-65 vs over-65 discrimination gap
(0.616 vs 0.667, descriptive) reflects a genuine age-dependent effect or
sampling variability, by fitting a logistic model with each of the three
final standardised features, standardised age, and their three product
terms, in the matched cohort (seed 42).

Result reported in the manuscript: none of the three interaction terms
approached significance (all p>0.38), while all three main effects
remained significant (p<=0.001) -- arguing against a genuine interaction
and against an earlier descriptive (feature-median) observation that the
RMSSD gap widens with age, which is accordingly withdrawn.

Input : cohort_clean.csv (seed-42 matched cohort)
Output: printed statsmodels Logit summary table
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

FEATURES = ["RMSSD", "PR_med", "Pamp_V1"]


def main():
    m = pd.read_csv("cohort_clean.csv")
    m["Pamp_V1"] = m.get("Pamp_6")
    m["y"] = (m["split"] == "vaka").astype(int)

    X = StandardScaler().fit_transform(m[FEATURES].fillna(m[FEATURES].median()))
    age_z = StandardScaler().fit_transform(m[["age_at_index"]])[:, 0]

    Xd = pd.DataFrame(X, columns=FEATURES)
    Xd["age_z"] = age_z
    for f in FEATURES:
        Xd[f + "_x_age"] = Xd[f] * Xd["age_z"]
    Xd = sm.add_constant(Xd)

    model = sm.Logit(m["y"], Xd).fit(disp=0)
    print(model.summary2().tables[1].round(4))


if __name__ == "__main__":
    main()
