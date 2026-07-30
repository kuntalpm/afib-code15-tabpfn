"""
Stage 4 - Atrial conduction and depolarisation features (Methods 2.4, 2.5).

Extracts, per record, across all 12 leads:
  - P duration, P dispersion (later classified as measurement artefact;
    Methods 2.5, kept here for the artefact-adjudication figure/table only)
  - PR interval (median-of-medians across beats then leads; Methods 2.4)
  - P-wave peak amplitude in V1 (final retained feature, Pamp_V1; index 6)
  - a provisional 40 ms pre-P-offset signed integral in V1 (the pre-final
    proxy later found to track P-wave amplitude, rho=0.85 with Pamp_V1;
    kept here only to reproduce the artefact-adjudication analysis, Methods
    2.4-2.5; NOT used in the final model)

V1 is lead index 6 in the CODE-15% 12-lead ordering
(I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6).

Input : cohort_signals.npz, signal_bounds.npz
Output: pwave_features.csv
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
FS = 400
V1 = 6


def main():
    sig = np.load("cohort_signals.npz")
    tracings, ids = sig["tracings"], sig["exam_id"]
    bounds = np.load("signal_bounds.npz")
    bound_map = dict(zip(bounds["exam_id"], zip(bounds["start"], bounds["end"])))
    idx_of = {e: i for i, e in enumerate(ids)}

    bb, ba = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    rows = []
    for j, exam_id in enumerate(ids):
        if j % 300 == 0:
            print(f"  {j}/{len(ids)}")
        s, e = bound_map[exam_id]
        if s < 0:
            continue
        sig12 = tracings[idx_of[exam_id]][s : e + 1, :]
        rec = {"exam_id": exam_id}
        p_durations, pr_intervals = [], []

        for lead in range(12):
            try:
                x = filtfilt(bb, ba, sig12[:, lead])
                _, info = nk.ecg_peaks(x, sampling_rate=FS)
                rp = info["ECG_R_Peaks"]
                if len(rp) < 4:
                    continue
                _, waves = nk.ecg_delineate(x, rp, sampling_rate=FS, method="dwt")
                on = np.array(waves["ECG_P_Onsets"], float)
                off = np.array(waves["ECG_P_Offsets"], float)
                r_on = np.array(waves["ECG_R_Onsets"], float)
                pk = np.array(waves["ECG_P_Peaks"], float)

                valid = ~(np.isnan(on) | np.isnan(off))
                if valid.sum() >= 2:
                    dur = float(np.median(off[valid] - on[valid])) / FS * 1000
                    if 40 < dur < 250:
                        p_durations.append(dur)
                        rec[f"Pdur_{lead}"] = dur

                pr_valid = ~(np.isnan(on) | np.isnan(r_on))
                if pr_valid.sum() >= 2:
                    pr = float(np.median(r_on[pr_valid] - on[pr_valid])) / FS * 1000
                    if 60 < pr < 400:
                        pr_intervals.append(pr)

                amp_valid = ~np.isnan(pk)
                if amp_valid.sum() >= 2:
                    rec[f"Pamp_{lead}"] = float(np.median(x[pk[amp_valid].astype(int)]))

                if lead == V1 and valid.sum() >= 2:
                    # provisional 40 ms pre-offset integral (superseded proxy)
                    seg = [
                        x[max(0, int(o) - 16) : int(o)]
                        for o in off[valid]
                        if int(o) > 16
                    ]
                    if seg:
                        rec["PTFV1_proxy"] = float(
                            np.median([t.sum() / FS * 1000 for t in seg])
                        )
            except Exception:
                continue

        rec["n_lead_ok"] = len(p_durations)
        if p_durations:
            rec["Pdur_max"] = max(p_durations)
            rec["Pdur_min"] = min(p_durations)
            rec["Pdisp"] = max(p_durations) - min(p_durations)
        if pr_intervals:
            rec["PR_med"] = float(np.median(pr_intervals))
            rec["PR_max"] = max(pr_intervals)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["Pamp_V1"] = df.get("Pamp_6")  # final retained feature
    df.to_csv("pwave_features.csv", index=False)
    print(f"\n-> pwave_features.csv ({len(df)} records)")
    print(f"PR_med coverage   : {df.PR_med.notna().sum()}/{len(df)}")
    print(f"Pamp_V1 coverage  : {df.Pamp_V1.notna().sum()}/{len(df)}")


if __name__ == "__main__":
    main()
