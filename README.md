# How much AF substrate is measurable on routine sinus-rhythm ECG?

Analysis code for the manuscript:

> **How much atrial fibrillation substrate is measurable on routine sinus-rhythm ECG? A leakage-free case-control analysis with age-ordered index recordings**

This repository contains the complete analysis workflow used to generate the final reported results, tables, and figures.

## Repository purpose

The study quantifies how much information about subsequently documented atrial fibrillation (AF) is recoverable from directly measurable features of routine 12-lead ECGs recorded during sinus rhythm.

The primary analysis uses a leakage-aware, patient-level design with age-ordered index recordings, age- and sex-matched controls, repeated group-stratified cross-validation, and seed-averaged matching. The repository also includes:

- ECG signal extraction and preprocessing
- RR-interval and P-wave feature extraction
- Measurement validation against device labels
- Positive and negative leakage sentinel analyses
- Matching and repeated cross-validation
- Seven-model comparison, including TabPFN
- PTB-XL replication analyses
- Short-window RMSSD validity analysis using the PAF Prediction Challenge Database
- Generation of manuscript tables and figures

## Data sources

The datasets are not redistributed in this repository.

| Source | Access | Used for |
|---|---|---|
| [CODE-15%](https://doi.org/10.5281/zenodo.4916206) | Open, Zenodo, CC-BY | Primary derivation cohort |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | Open, PhysioNet, ODC-BY | External replication |
| [PAF Prediction Challenge Database (afpdb)](https://physionet.org/content/afpdb/) | Open, PhysioNet | Short-window RMSSD validity substudy |

Download each dataset directly from its original source and place the raw files in a local `data/` directory. Paths are defined at the top of the relevant scripts and must be adapted to the local machine.

Do not upload, redistribute, or commit raw ECG data or patient-level derived files unless this is permitted by the applicable dataset licence and data-use terms.

## Requirements

The final analysis was run using Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Exact package versions used for the reported results are pinned in `requirements.txt` and recorded in `config/package_versions.txt`.

The TabPFN analyses use package version 8.1.0, which loads the TabPFN-3 default classification checkpoint:

```text
tabpfn-v3-classifier-v3_default.ckpt
```

The TabPFN checkpoint is licensed for research and evaluation use, not for commercial or production use. See the manuscript Data and Code Availability statement and the relevant TabPFN licence for details.

## Reproducing the analysis

Scripts are numbered and should be run sequentially from the repository root. Each stage writes intermediate outputs required by later stages.

```bash
python src/data/00_verify_af_label.py
python src/data/01_build_cohort.py
python src/data/02_extract_signals.py
python src/features/03_rr_features.py
python src/features/04_pwave_features.py
python src/features/05_ptfv1_classic.py
python src/validation/06_leakage_sentinels.py
python src/matching/07_seed_averaged_matching.py
python src/models/08_model_comparison.py
python src/validation/09_ptbxl_replication.py
python src/validation/10_s1_rmssd_validity.py
python src/models/11_age_interaction.py
```

Generate manuscript figures after the analysis stages:

```bash
python src/figures/[figure_script_name].py
```

Waveform processing, repeated cross-validation, bootstrap analyses, and external replication may require substantial runtime. The complete workflow should be run in a clean environment after all three source datasets have been downloaded.

## Analysis pipeline

| Stage | Script | Main output | Manuscript location |
|---|---|---|---|
| 0 | `src/data/00_verify_af_label.py` | Cohort feasibility count, G0 | Methods 2.2 |
| 1 | `src/data/01_build_cohort.py` | `cohort_clean.csv` | Methods 2.2; Figure 1 |
| 2 | `src/data/02_extract_signals.py` | `cohort_signals.npz`, `signal_bounds.npz` | Methods 2.3 |
| 3 | `src/features/03_rr_features.py` | `rr_features.csv` | Methods 2.3 and 2.6 |
| 4 | `src/features/04_pwave_features.py` | `pwave_tam.csv` | Methods 2.4 and 2.5 |
| 5 | `src/features/05_ptfv1_classic.py` | `ptfv1_true.csv` | Methods 2.4; Results 3.2 |
| 6 | `src/validation/06_leakage_sentinels.py` | Sentinel-1 and Sentinel-2 AUROC | Methods 2.7; Results |
| 7 | `src/matching/07_seed_averaged_matching.py` | Seed-averaged AUROC; H1/H2 analyses | Methods 2.9; Results 3.3–3.7 |
| 8 | `src/models/08_model_comparison.py` | Seven-model comparison; Table 3 | Results 3.3 and 3.6 |
| 9 | `src/validation/09_ptbxl_replication.py` | PTB-XL feature replication and retrained discrimination | Results 3.11 |
| 10 | `src/validation/10_s1_rmssd_validity.py` | ICC and Bland–Altman analysis; Table 7 | Methods 2.10; Results 3.10 |
| 11 | `src/models/11_age_interaction.py` | Age-by-feature interaction analysis | Discussion |
| 12 | `src/figures/*.py` | Figures 1–6 | Figure legends |

## Reproducibility controls

- `config/seeds.txt` contains the 11 matching seeds used for seed-averaged estimates.
- `config/package_versions.txt` records the computational environment used for the final results.
- Cross-validation partitioning and bootstrap analyses use fixed seeds defined in the repository configuration.
- All preprocessing that can introduce information leakage is fitted within training folds in the relevant model scripts.
- The repository is intended to reproduce the final reported analyses from the cited source data, subject to software, hardware, and upstream dataset-version differences.

Minor numerical variation may occur across operating systems or numerical-library implementations. Such variation should not alter the principal conclusions; material discrepancies should be investigated before interpreting results.

## Included and excluded materials

### Included

This repository includes every final analysis script needed to proceed from the downloaded source data to the numerical results, tables, and figures reported in the manuscript.

### Not included

The repository does not include:

- Raw source datasets or patient-level data
- Exploratory scripts that were not used for the final reported analyses
- Intermediate development variants that were superseded by the final pipeline
- Private local paths, credentials, temporary files, or cached model artefacts

Two superseded methodological variants are discussed transparently in the manuscript but are not required for reproducing the final results:

- The original asymmetric control-selection rule, which produced a spurious age difference between cases and controls
- The pre-final PTFV1 proxy measure, later found to track P-wave amplitude

These are documented in Methods 2.2 and Methods 2.4–2.5 as part of the methodological development trail.

## Generative-AI assistance and author responsibility

Claude Code (Anthropic; model/version: `[insert model and version if available]`) assisted with drafting and refactoring Python analysis scripts, implementing statistical analyses, and drafting or revising manuscript text.

The authors defined the research question, scientific objectives, cohort definitions, analytic requirements, and interpretation framework. The authors executed the code locally, reviewed and revised the implementation, checked the analytic outputs and reported results, and retain full responsibility for the study design, data handling, methods, analyses, results, interpretations, references, originality, and integrity of the work.

No generative-AI system is an author of this repository or the associated manuscript.

## Citation

If you use this code, please cite the associated manuscript once it is published:

```text
[Add full manuscript citation and DOI after publication]
```

The repository should be archived as a versioned release before submission, for example:

```text
v1.0-submission
```

A persistent Zenodo DOI for the archived repository release should be added here when available.

## Licence

The analysis code in this repository is released under the MIT License; see `LICENSE`.

This licence applies only to the code in this repository. It does not apply to:

- CODE-15%
- PTB-XL
- The PAF Prediction Challenge Database
- TabPFN model weights or checkpoints
- Any other third-party software, data, or model components

Users are responsible for complying with the licences and data-use conditions of all third-party resources.
