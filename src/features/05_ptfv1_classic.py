"""
Stage 5 - Classic (sign-aware) P-terminal force in V1 (Morris index; Results
3.2, Table 2).

Unlike the provisional 40 ms integral in 04_pwave_features.py (which
integrates the raw filtered signal regardless of sign and was later found to
proxy P-wave amplitude), this computes the true Morris index: the signed
area of the terminal NEGATIVE deflection within the P wave, found by
scanning backwards from P offset to the last negative-to-positive crossing.

This measure does not differ significantly between cases and controls
(Results 3.2; Hodges-Lehmann difference +0.02, 95% CI 0.00 to +0.06) and is
reported for completeness and to contrast with the artefact identified in
04_pwave_features.py's proxy integral.

Input : cohort_signals.npz, signal_bounds.npz
Output: ptfv1_true.csv
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
FS = 400
V1 = 6


def terminal_negative_area(segment, fs):
    """Signed area (ms-equivalent units) of the terminal negative deflection
    within a P-wave segment, scanning backwards from the end."""
    if len(segment) == 0 or not (segment < 0).any():
        return 0.0
    k = len(segment) - 1
    while k >= 0 and segment[k] >= 0:
        k -= 1
    j = k
    while j >= 0 and segment[j] < 0:
        j -= 1
    term = segment[j + 1 : k + 1]
    if len(term) == 0:
        return 0.0
    return float(term.sum() / fs * 1000)


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
        x = filtfilt(bb, ba, tracings[idx_of[exam_id]][s : e + 1, V1])
        try:
            _, info = nk.ecg_peaks(x, sampling_rate=FS)
            rp = info["ECG_R_Peaks"]
            if len(rp) < 4:
                continue
            _, waves = nk.ecg_delineate(x, rp, sampling_rate=FS, method="dwt")
            on = np.array(waves["ECG_P_Onsets"], float)
            off = np.array(waves["ECG_P_Offsets"], float)
            valid = ~(np.isnan(on) | np.isnan(off))
            vals = []
            for o1, o2 in zip(on[valid].astype(int), off[valid].astype(int)):
                if o2 <= o1 or o2 > len(x):
                    continue
                vals.append(terminal_negative_area(x[o1:o2], FS))
            if vals:
                rows.append({"exam_id": exam_id, "PTFV1_true": float(np.median(vals))})
        except Exception:
            continue

    pd.DataFrame(rows).to_csv("ptfv1_true.csv", index=False)
    print(f"\n-> ptfv1_true.csv ({len(rows)} records)")


if __name__ == "__main__":
    main()
