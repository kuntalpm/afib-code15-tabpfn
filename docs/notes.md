# Methodological decision log

Brief notes on decisions that are easy to get wrong when re-implementing
this pipeline, kept here rather than only in the manuscript prose.

**Zero-padding.** CODE-15% records shorter than 10.24 s are zero-padded to
4,096 samples. Strip padding (detect the first/last non-zero sample) before
any filtering. Filtering across a zero/signal boundary produces edge
artefacts and can be misread as asystole downstream.

**Sinus criterion.** Use `AF == 0`, not the composite `normal_ecg` flag.
The composite flag also requires absence of first-degree AV block, sinus
bradycardia and sinus tachycardia, which excludes exactly the atrial-
conduction phenotype under study. Confirmed empirically: `AF == 0` yields
618 candidate cases vs 135 under the stricter flag.

**Control selection.** Earliest eligible record per control patient, fixed
a priori. Confirmed empirically against the full control cohort rather than
assumed.

**PTFV1 is two different things in this codebase.** `04_pwave_features.py`
computes a provisional 40 ms pre-P-offset signed integral (used only for
the artefact-adjudication analysis). `05_ptfv1_classic.py` computes the
true Morris index (signed area of the terminal negative deflection only).
The provisional integral turned out to correlate at rho=0.85 with P-wave
amplitude in V1 and was replaced by amplitude in the final model; it is
not the classic P-terminal force and should not be reported as such. This
was a real error caught during external review of the manuscript and is
documented here so it is not repeated.

**Matching is random -- average over it.** A single matching realisation
can materially inflate or deflate the headline AUROC (in this dataset, by
about 0.015-0.03 depending on the quantity). All primary estimates are
averaged over 11 independent realisations (`config/seeds.txt`); a single
realisation (seed 42) is retained only for tables/figures that require one
fixed dataset to display, and this is stated wherever it applies.

**Feature reduction has two stages, and only one is safe.** Stage 1
(technical-descriptor removal, collinearity reduction) uses no outcome
information and can be done outside the CV loop. Stage 2 (dropping mean RR
and max PR by permutation importance, substituting P-wave amplitude for
the provisional integral) uses outcome information and was done outside
the CV loop; this carries a residual optimism risk, quantified in the
manuscript by comparing against an outcome-blind in-loop selection (which
selects diferently and worse: it prefers recording-duration and RR-count,
properties of the acquisition rather than the patient).

**Table numbering.** Tables are numbered in order of first citation in the
text, not in the order they were written. This was wrong in an earlier
draft (twice) and caught by external review; if you add or remove a table,
re-check the citation order.
