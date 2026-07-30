"""
Stage 10 - Short-window RMSSD measurement validity (Methods 2.10, Results
3.10, Table 7).

Uses the PAF Prediction Challenge Database (afpdb; PhysioNet/CinC 2001,
open access, 128 Hz, 30-minute two-lead recordings) to test whether RMSSD
computed over ~8 RR intervals (the window available in a 7-10 s ECG)
agrees with the conventional 5-minute reference.

Confidence intervals are cluster-robust: resampling whole RECORDINGS
(the independent unit, 100 total) rather than the 500 five-minute windows
they contain, since multiple windows from one recording are not
independent.

Input : afpdb *.dat/*.hea/*.qrs files (download separately, see README)
Output: s1_rmssd.csv, printed ICC(2,1)/ICC(3,1) with cluster-robust CIs
"""
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import warnings
import glob
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
FS = 128


def icc21(a, b):
    X = np.column_stack([a, b])
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def icc31(a, b):
    X = np.column_stack([a, b])
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


def extract_windows():
    bb, ba = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    rows = []
    for rec_path in sorted(glob.glob("data/afpdb/*.dat")):
        rec = rec_path[:-4]
        group = "PAF" if rec.split("/")[-1][0] == "p" else "kontrol"
        try:
            sig, _ = wfdb.rdsamp(rec)
            x = filtfilt(bb, ba, sig[:, 0])
            _, info = nk.ecg_peaks(x, sampling_rate=FS)
            rp = info["ECG_R_Peaks"]
            rr = np.diff(rp) / FS * 1000
            rr = rr[(rr > 300) & (rr < 2000)]
            if len(rr) < 200:
                continue
            t = np.cumsum(rr) / 1000
            for w0 in np.arange(0, max(t[-1] - 300, 0), 300):
                sel = (t >= w0) & (t < w0 + 300)
                rr5 = rr[sel]
                if len(rr5) < 200:
                    continue
                gold = float(np.sqrt(np.mean(np.diff(rr5) ** 2)))
                t5 = t[sel] - w0
                short = []
                for s0 in np.arange(0, 290, 10):
                    m = (t5 >= s0) & (t5 < s0 + 10)
                    rk = rr5[m]
                    if len(rk) >= 6:
                        short.append(float(np.sqrt(np.mean(np.diff(rk) ** 2))))
                if len(short) >= 15:
                    rows.append({
                        "rec": rec, "group": group,
                        "gold": gold, "short_median": float(np.median(short)),
                    })
        except Exception:
            continue
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(sub, n_boot=2000, seed=0):
    recs = sub.rec.unique()
    rng = np.random.default_rng(seed)
    icc2, icc3 = [], []
    for _ in range(n_boot):
        chosen = rng.choice(recs, len(recs), replace=True)
        rows = pd.concat([sub[sub.rec == r] for r in chosen], ignore_index=True)
        a, b = rows.gold.values, rows.short_median.values
        if len(a) < 10:
            continue
        icc2.append(icc21(a, b))
        icc3.append(icc31(a, b))
    return np.percentile(icc2, [2.5, 97.5]), np.percentile(icc3, [2.5, 97.5])


def main():
    df = extract_windows()
    df.to_csv("s1_rmssd.csv", index=False)

    for label, sub in [("All", df), ("PAF", df[df.group == "PAF"]),
                        ("Control", df[df.group == "kontrol"])]:
        a, b = sub.gold.values, sub.short_median.values
        ci2, ci3 = cluster_bootstrap_ci(sub)
        print(f"{label} (n={len(sub)} windows, {sub.rec.nunique()} recordings)")
        print(f"  ICC(2,1) {icc21(a,b):.3f}  cluster-CI [{ci2[0]:.3f},{ci2[1]:.3f}]")
        print(f"  ICC(3,1) {icc31(a,b):.3f}  cluster-CI [{ci3[0]:.3f},{ci3[1]:.3f}]")
        print(f"  median ratio (short/gold): {np.median(b/a):.2f}\n")


if __name__ == "__main__":
    main()
