# How much AF substrate is measurable on routine sinus-rhythm ECG?

Analysis code for the manuscript *"How much atrial fibrillation substrate is
measurable on routine sinus-rhythm ECG? A leakage-free case-control analysis
with age-ordered index recordings."*

## Data sources (not included in this repository)

| Source | Access | Used for |
|---|---|---|
| [CODE-15%](https://doi.org/10.5281/zenodo.4916206) | Open (Zenodo, CC-BY) | Primary derivation cohort |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | Open (PhysioNet, ODC-BY) | External replication |
| [PAF Prediction Challenge Database (afpdb)](https://physionet.org/content/afpdb/) | Open (PhysioNet) | Short-window RMSSD validity substudy |

None of these datasets are redistributed here. Each script expects the raw
files to be downloaded into a local `data/` directory (paths are set at the
top of each script and should be adjusted to your machine).

## Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Exact versions used to produce the numbers reported in the manuscript are
pinned in `requirements.txt`. TabPFN loads the TabPFN-3 checkpoint
(`tabpfn-v3-classifier-v3_default.ckpt`) under package version 8.1.0; this
checkpoint is licensed for research and evaluation, not commercial or
production use (see manuscript Data and Code Availability statement).

## Pipeline order

Scripts are numbered in the order they are run. Each stage writes an
intermediate file consumed by the next.

| Stage | Script | Produces | Manuscript section |
|---|---|---|---|
| 0 | `src/data/00_verify_af_label.py` | Cohort feasibility count (G0) | Methods 2.2 |
| 1 | `src/data/01_build_cohort.py` | `cohort_clean.csv` | Methods 2.2, Fig. 1 |
| 2 | `src/data/02_extract_signals.py` | `cohort_signals.npz`, `signal_bounds.npz` | Methods 2.3 |
| 3 | `src/features/03_rr_features.py` | `rr_features.csv` | Methods 2.3, 2.6 |
| 4 | `src/features/04_pwave_features.py` | `pwave_tam.csv` | Methods 2.4, 2.5 |
| 5 | `src/features/05_ptfv1_classic.py` | `ptfv1_true.csv` | Methods 2.4, Results 3.2 |
| 6 | `src/validation/06_leakage_sentinels.py` | Sentinel-1 / Sentinel-2 AUROC | Methods 2.7, Results |
| 7 | `src/matching/07_seed_averaged_matching.py` | Seed-averaged AUROC, H1/H2 | Methods 2.9, Results 3.3-3.7 |
| 8 | `src/models/08_model_comparison.py` | 7-model comparison (Table 3) | Results 3.3 |
| 9 | `src/validation/09_ptbxl_replication.py` | External replication (proxy + true amplitude) | Results 3.11 |
| 10 | `src/validation/10_s1_rmssd_validity.py` | ICC / Bland-Altman (Table 7) | Methods 2.10, Results 3.10 |
| 11 | `src/models/11_age_interaction.py` | Age-by-feature interaction test | Discussion §4 |
| 12 | `src/figures/*.py` | Figures 1-6 | Figure legends |

`config/seeds.txt` lists the 11 matching seeds averaged over for all
seed-averaged estimates reported in the manuscript. `config/package_versions.txt`
records the exact environment used for the reported numbers.

## What is and is not included

Included: every script needed to go from raw downloaded data to every number,
table and figure in the manuscript, in the form actually used for the final
reported results.

Not included: the many intermediate/exploratory variants run during method
development (e.g. the original mis-specified control-selection rule that
produced a spurious age gap, or the pre-final PTFV1 proxy later found to track
P-wave amplitude). These are described, not concealed: both are discussed in
the manuscript itself (Methods 2.2, Methods 2.4-2.5) as part of the recorded
methodological trail, and are not needed to reproduce the final numbers.

## Citing

See the manuscript reference list. If you use this code, please cite the
manuscript once published; a Zenodo DOI for this repository will be added on
acceptance.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
This licensing covers the analysis code only, not the third-party datasets or
the TabPFN model weights, which carry their own licenses noted above.
