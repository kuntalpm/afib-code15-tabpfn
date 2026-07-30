"""
Stage 8 - Seven model classes on the primary five-feature set (Methods 2.8,
Table 3, Results 3.3).

Primary five-feature set (outcome-blind, Methods 2.4 Stage 1):
  meanRR, RMSSD, PR_med, PR_max, and the provisional V1 integral
  (PTFV1_proxy) -- see 04_pwave_features.py. This is the set the primary
  headline AUROC (0.653, seed-averaged) refers to.

Logistic regression, SVM, kNN and random forest are run at library
defaults. Gradient boosting and the multilayer perceptron are tuned by
3-fold grid search nested within each training fold. TabPFN is run with
package defaults and no hyperparameter search (package version and
checkpoint recorded in config/package_versions.txt).

Input : cohort_clean.csv (seed-42 matched cohort from stage 7)
Output: printed AUROC for all seven models with 95% bootstrap CI
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
FEATURES5 = ["meanRR", "RMSSD", "PR_med", "PR_max", "PTFV1_proxy"]


def oof(m_frame, features, model_fn, n_repeats=10, scale=True):
    y = (m_frame["split"] == "vaka").astype(int).values
    X = SimpleImputer().fit_transform(m_frame[features].values)
    P = np.zeros((n_repeats, len(y)))
    for t in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=t)
        for tr, te in skf.split(X, y, groups=m_frame.patient_id.values):
            model = make_pipeline(StandardScaler(), model_fn()) if scale else model_fn()
            model.fit(X[tr], y[tr])
            P[t, te] = model.predict_proba(X[te])[:, 1]
    return y, P.mean(0)


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            boots.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(boots, [2.5, 97.5])


def main():
    m = pd.read_csv("cohort_clean.csv")

    def gbdt():
        return GridSearchCV(
            HistGradientBoostingClassifier(random_state=42),
            {"max_leaf_nodes": [4, 8, 15], "learning_rate": [0.03, 0.1],
             "min_samples_leaf": [20, 50]},
            scoring="roc_auc", cv=3, n_jobs=-1,
        )

    def mlp():
        return GridSearchCV(
            MLPClassifier(max_iter=2000, random_state=42),
            {"hidden_layer_sizes": [(16,), (32,), (64, 32)],
             "alpha": [0.1, 1.0, 10.0]},
            scoring="roc_auc", cv=3, n_jobs=-1,
        )

    models = {
        "Logistic regression": (lambda: LogisticRegression(max_iter=2000), True, 10),
        "SVM (RBF)": (lambda: SVC(probability=True, random_state=42), True, 10),
        "k-NN (k=50)": (lambda: KNeighborsClassifier(50), True, 10),
        "Random forest": (lambda: RandomForestClassifier(400, random_state=42), True, 10),
        "Gradient boosting (tuned)": (gbdt, True, 3),
        "MLP (tuned)": (mlp, True, 3),
    }

    print("=== Seven model classes, primary 5-feature set ===")
    for name, (fn, scale, reps) in models.items():
        y, p = oof(m, FEATURES5, fn, n_repeats=reps, scale=scale)
        auc, ci = bootstrap_ci(y, p)
        print(f"  {name:28s} {auc:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]")

    try:
        from tabpfn import TabPFNClassifier
        y, p = oof(m, FEATURES5, lambda: TabPFNClassifier(), n_repeats=5, scale=False)
        auc, ci = bootstrap_ci(y, p)
        print(f"  {'TabPFN-3':28s} {auc:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]")
    except ImportError:
        print("  TabPFN not installed (pip install tabpfn)")


if __name__ == "__main__":
    main()
