# LEAD-EEG: A Lightweight Ensemble Framework for Alzheimer's Detection from EEG

[![tests](https://github.com/aftab802/lead-eeg/actions/workflows/tests.yml/badge.svg)](https://github.com/aftab802/lead-eeg/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Energy, power and spectral-entropy biomarkers, polynomial expansion, fold-local SelectKBest and SMOTE, and an RF + GB soft-voting ensemble.

Code accompanying the manuscript **“LEAD-EEG: A Lightweight Ensemble Framework for Alzheimer's Detection from EEG”** by Muhammad Aftab ur Rehman, Muhammad Awais ur Rehman, Muhammad Mohsin Sajjad (submitted, 2026).
Corresponding author: Muhammad Aftab ur Rehman, CYBEX – School of IT Professionals (Pvt) Limited, Faisalabad, Pakistan.

## Installation

```bash
git clone https://github.com/aftab802/lead-eeg.git
cd lead-eeg
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10–3.13 (tested on 3.13 with TensorFlow 2.20). A GPU is optional.

## Data

OpenNeuro **ds004504** v1.0.6 — resting-state, eyes-closed EEG of 88 subjects (36 AD, 23 FTD, 29 HC), 19 channels,
500 Hz — https://openneuro.org/datasets/ds004504 (Miltiadous et al., *Data* 8(6):95, 2023,
doi [10.3390/data8060095](https://doi.org/10.3390/data8060095)). Download it, for example, with

```bash
pip install openneuro-py
openneuro-py download --dataset ds004504 --target-dir data/ds004504
```

The scripts read the denoised recordings in `derivatives/sub-XXX/eeg/` and labels from `participants.tsv`.

## Reproducing the paper

```bash
# LEAD-EEG cohort: CSV with columns path (any format mne.io.read_raw opens) and label (1 dementia, 0 healthy)
python scripts/run_lead_eeg.py --labels lead_eeg_labels.csv
# public proxy until the cohort accession is released (not the paper's cohort)
python scripts/run_lead_eeg.py --ds004504 data/ds004504
```

Every run writes a JSON file to `results/` with the test metrics, confusion matrix, split sizes and, for subject
data, `train_test_subject_overlap` (0 under the leakage-safe protocol).

## Reproduction status

| Result in the paper | Value | Source of the number | Re-run with this repository? |
|---|---|---|---|
| 10-fold accuracy | 76.14 % | Original run; cohort accession pending | Needs the 176-subject cohort. Public proxy (ds004504 AD vs HC, 65 subjects): **61.54 %** |
| Dementia precision / recall | 0.81 / 0.84 | Same | Reported in the result JSON |

## Points to resolve against the manuscript

* With three channel-averaged biomarkers, selecting k = 25 features requires a polynomial degree of at least 4
  (34 expanded features). The default here computes the biomarkers per channel and uses degree 2; state the
  configuration actually used in the paper.
* Random-forest and gradient-boosting settings are not given in the paper; defaults used here are 300 and 100 trees.
* Recordings are read lazily (`preload=False`), one 4-second epoch at a time.

## Repository layout

```
cybex_eeg/        shared, leakage-aware library (splits, normalisation, features, models)
scripts/          one command per experiment in the paper; results are written to results/*.json
tests/            pytest suite run on synthetic data (no download needed)
```

## Tests

```bash
pytest -q
```
Tests check the identifier-keyed label merge (and that a positional merge misassigns labels), the 28 qEEG and 11 graph features, the LEAD-EEG biomarkers, and that feature selection and SMOTE are fitted inside training folds only.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the article is published, please cite the manuscript title above and
this repository.

## Related repositories

* [eeg-mi-leakage-benchmark](https://github.com/aftab802/eeg-mi-leakage-benchmark) — A Systematic Benchmark of Motor Imagery Classifiers Under Leakage-Safe Evaluation
* [eeg-dementia-integrity-benchmark](https://github.com/aftab802/eeg-dementia-integrity-benchmark) — Data Integrity and Feature Engineering in Small-Cohort EEG Dementia Classification
* [eeg-mi-protocol-leakage-case-study](https://github.com/aftab802/eeg-mi-protocol-leakage-case-study) — A Case Study of Evaluation-Protocol Leakage in Motor Imagery EEG Classification
* [eeg-mi-four-class-cnn](https://github.com/aftab802/eeg-mi-four-class-cnn) — Four-Class Motor Imagery Classification Using a Residual Convolutional Neural Network
* [eeg-mi-fbcsp-cnn](https://github.com/aftab802/eeg-mi-fbcsp-cnn) — CSP+CNN: Filter-Bank Common Spatial Patterns with a Convolutional Classifier for Motor Imagery Classification from EEG
* [group-aware-eegnet](https://github.com/aftab802/group-aware-eegnet) — Group-Aware EEGNet: Subject-Independent Motor Imagery Classification from EEG
* [eeg-dementia-label-correction](https://github.com/aftab802/eeg-dementia-label-correction) — Label Correction and Spectral Features for Multiclass Dementia Detection from EEG

## License

MIT — see [LICENSE](LICENSE). The datasets are distributed by their owners under their own terms.
