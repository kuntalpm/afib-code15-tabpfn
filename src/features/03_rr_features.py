"""
Stage 3 - Beat-to-beat variability features (Methods 2.3, 2.6).

R-peak detection on lead II (index 1) after zero-padding removal and
0.5-40 Hz zero-phase band-pass filtering. RR intervals outside 300-2,000 ms
are discarded as physiologically implausible.

Signal quality gate: >=6 RR intervals required (relaxed from an initial
>=8 R-peak criterion, which was found to reject records on the basis of
heart rate rather than signal quality -- at 7.33 s, 8 beats requires a rate
above ~65/min. Of 609 records failing the original criterion, 567 had 6-7
detected beats and only 6 had 3 or fewer; see Methods 2.3).

Validated against device labels (Methods 2.6): heart rate derived here
separates device-labelled sinus bradycardia / tachycardia with AUROC
0.944 / 0.984 respectively.

Input : cohort_signals.npz, signal_bounds.npz
Output: rr_features.csv
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
FS = 400  # CODE-15% native sampling rate
LEAD_II = 1


def main():
    sig = np.load("cohort_signals.npz")
    tracings, ids = sig["tracings"], sig["exam_id"]
    bounds = np.load("signal_bounds.npz")
    bound_map = dict(zip(bounds["exam_id"], zip(bounds["start"], bounds["end"])))
    idx_of = {e: i for i, e in enumerate(ids)}

    bb, ba = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    rows = []
    for j, exam_id in enumerate(ids):
        if j % 500 == 0:
            print(f"  {j}/{len(ids)}")
        s, e = bound_map[exam_id]
        if s < 0:
            continue
        sig12 = tracings[idx_of[exam_id]]
        x = filtfilt(bb, ba, sig12[s : e + 1, LEAD_II])
        try:
            _, info = nk.ecg_peaks(x, sampling_rate=FS)
            rp = info["ECG_R_Peaks"]
        except Exception:
            rp = np.array([])
        rr = np.diff(rp) / FS * 1000
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 6:
            continue
        rows.append({
            "exam_id": exam_id,
            "meanRR": float(rr.mean()),
            "SDNN": float(rr.std(ddof=1)) if len(rr) > 1 else np.nan,
            "RMSSD": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else np.nan,
            "n_RR": int(len(rr)),
            "n_R": int(len(rp)),
            "dur_s": float((e - s + 1) / FS),
        })

    pd.DataFrame(rows).to_csv("rr_features.csv", index=False)
    print(f"\n-> rr_features.csv ({len(rows)} records)")


if __name__ == "__main__":
    main()
