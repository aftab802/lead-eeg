"""LEAD-EEG: biomarkers + polynomial expansion + SelectKBest(25) + SMOTE +
RF/GB soft voting, evaluated with 10-fold stratified CV (all fitted steps
inside the folds).

Cohort input (176-subject LEAD-EEG cohort): --labels CSV with columns
'path' (EEG file readable by mne.io.read_raw) and 'label' (1 = dementia,
0 = healthy). Public proxy: --ds004504 BIDS_ROOT runs the same pipeline on the
AD vs HC subset of OpenNeuro ds004504; this is NOT the paper's cohort.
"""
import numpy as np
import pandas as pd
from _common import parser, save

from cybex_eeg.dementia import ds004504 as D
from cybex_eeg.dementia.lead import evaluate, subject_vector

if __name__ == "__main__":
    import mne

    p = parser(__doc__)
    p.add_argument("--labels", help="CSV with columns path,label")
    p.add_argument("--ds004504", help="BIDS root for the public AD-vs-HC proxy run")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--channel-average", action="store_true", help="3 base features instead of 3 per channel")
    a = p.parse_args()
    if a.labels:
        df = pd.read_csv(a.labels)
    elif a.ds004504:
        parts = D.load_participants(a.ds004504).set_index("subject_id")
        rows = [(path, 1 if parts.loc[sub, "Group"] == "A" else 0) for sub, path in D.derivative_recordings(a.ds004504)
                if parts.loc[sub, "Group"] in ("A", "C")]
        df = pd.DataFrame(rows, columns=["path", "label"])
    else:
        p.error("give --labels or --ds004504")
    X = []
    for path in df["path"]:
        raw = mne.io.read_raw(path, preload=False, verbose=False)  # lazy: epochs read one at a time
        X.append(subject_vector(raw, per_channel=not a.channel_average))
    X, y = np.vstack(X), df["label"].to_numpy(int)
    print("Subjects:", len(y), "dementia:", int(y.sum()), "features:", X.shape[1])
    res = evaluate(X, y, n_splits=a.folds, k=a.k, degree=a.degree, random_state=a.seed)
    res.update(cohort="LEAD-EEG" if a.labels else "ds004504 AD-vs-HC proxy", n_subjects=int(len(y)))
    print(f"Pooled {a.folds}-fold accuracy: {res['accuracy_pooled']:.4f}")
    save(res, "lead_eeg", a.outdir)
