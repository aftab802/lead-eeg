# LEAD-EEG: A Lightweight Ensemble Framework for Dementia Detection from EEG

[![tests](https://github.com/aftab802/lead-eeg/actions/workflows/tests.yml/badge.svg)](https://github.com/aftab802/lead-eeg/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Energy, power and spectral-entropy biomarkers, polynomial expansion, fold-local SelectKBest and SMOTE, and an RF + GB soft-voting ensemble.

Code accompanying the manuscript **“LEAD-EEG: A Lightweight Ensemble Framework for Dementia Detection from EEG”** by Muhammad Aftab ur Rehman, Muhammad Awais ur Rehman, Muhammad Mohsin Sajjad (submitted, 2026).
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
# the paper's cohort: 176 recordings from 88 participants (ds004504 eyes-closed + ds006036 photic stimulation)
python scripts/run_lead_cohort176.py --ds004504 data/ds004504 --ds006036 data/ds006036
# any other cohort: CSV with columns path (any format mne.io.read_raw opens) and label (1 dementia, 0 healthy)
python scripts/run_lead_eeg.py --labels my_cohort.csv
```

Every run writes a JSON file to `results/` with the test metrics, confusion matrix, split sizes and, for subject
data, `train_test_subject_overlap` (0 under the leakage-safe protocol).

## Reproduction status

| Result in the paper | Value | Source of the number | Re-run with this repository? |
|---|---|---|---|
| 10-fold accuracy, recording-level folds | 76.14 % | Original run on 176 recordings (88 participants; ds004504 + ds006036) | **Re-run: 75.57 %** (`run_lead_cohort176.py`, CI job `cohort176`) |
| 10-fold accuracy, participant-grouped folds | 73.30 % | Re-run of this code | **Re-run: 73.30 % (fold mean 73.33 ± 5.99 %)** |
| Dementia precision / recall | 0.81 / 0.84 | Same | Reported in the result JSON |

## Points to resolve against the manuscript

* The cohort is 176 **recordings** from 88 participants (dementia = AD + FTD). Recording-level folds can split a
  participant's two recordings across training and test; the participant-grouped figure (73.30 %) is the leakage-safe one.
* Per-channel biomarkers (57 base features) with a degree-2 expansion reproduce the paper (75.57 %). Three
  channel-averaged biomarkers with the degree-4 expansion needed for k = 25 give 52.27 % (58.52 % grouped).
* ds006036 is downloaded like ds004504 (`openneuro-py download --dataset ds006036`); its denoised files are in `derivatives/eeglab/`.
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
