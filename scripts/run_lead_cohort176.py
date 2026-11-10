"""LEAD-EEG on the 176-recording dementia-vs-healthy cohort.

The cohort is built from two public OpenNeuro datasets recorded from the SAME
88 participants (Miltiadous et al. 2023; Ntetska et al. 2025):

  ds004504  eyes-closed resting EEG      (DOI 10.18112/openneuro.ds004504.v1.0.6)
  ds006036  eyes-open photic stimulation  (DOI 10.18112/openneuro.ds006036.v1.0.2)

Dementia = AD (36) + FTD (23) = 59 participants -> 118 recordings;
healthy  = cognitively normal (29)               ->  58 recordings.

Because every participant contributes two recordings, the script evaluates the
pipeline twice: (a) recording-level 10-fold CV, in which the two recordings of
one participant can fall on opposite sides of a fold boundary, and (b)
participant-grouped 10-fold CV (StratifiedGroupKFold), the leakage-safe estimate.
Both are run for the per-channel biomarker configuration (57 base features,
degree 2) and the channel-averaged configuration (3 base features, degree 4,
the smallest degree giving at least k = 25 expanded features).
"""
import numpy as np
import pandas as pd
from _common import parser, save

from cybex_eeg.dementia import ds004504 as D
from cybex_eeg.dementia.lead import evaluate, subject_vector


def cohort(ds004504: str, ds006036: str) -> pd.DataFrame:
    parts = D.load_participants(ds004504).set_index("subject_id")
    parts2 = D.load_participants(ds006036).set_index("subject_id")
    common = parts.index.intersection(parts2.index)
    if len(common) != len(parts) or (parts.loc[common, "Group"] != parts2.loc[common, "Group"]).any():
        raise D.LabelAlignmentError("ds004504 and ds006036 participant tables disagree")
    rows = []
    for name, root, task in (("eyes-closed", ds004504, "eyesclosed"), ("photic", ds006036, None)):
        for sub, path in D.derivative_recordings(root, task=task):
            g = parts.loc[sub, "Group"]
            rows.append(dict(participant=sub, recording=name, path=path, group=g, label=int(g in ("A", "F"))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import mne

    p = parser(__doc__)
    p.add_argument("--ds004504", required=True)
    p.add_argument("--ds006036", required=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--folds", type=int, default=10)
    a = p.parse_args()
    df = cohort(a.ds004504, a.ds006036)
    print("Recordings:", len(df), "participants:", df.participant.nunique(),
          "dementia recordings:", int(df.label.sum()), "healthy recordings:", int((1 - df.label).sum()))
    per_ch, avg = [], []
    for path in df.path:
        raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
        raw.pick([c for c in D.CHANNELS_19 if c in raw.ch_names])
        v = subject_vector(raw, per_channel=True)
        per_ch.append(v); avg.append(v.reshape(-1, 3).mean(axis=0))
    y = df.label.to_numpy(int)
    groups = df.participant.to_numpy()
    out = {"n_recordings": len(df), "n_participants": int(df.participant.nunique()),
           "n_dementia_recordings": int(y.sum()), "n_healthy_recordings": int(len(y) - y.sum()), "runs": {}}
    for cfg, X, degree in (("per_channel_deg2", np.vstack(per_ch), 2), ("channel_average_deg4", np.vstack(avg), 4)):
        for prot, g in (("recording_level", None), ("participant_grouped", groups)):
            r = evaluate(X, y, n_splits=a.folds, k=a.k, degree=degree, random_state=a.seed, groups=g)
            r["n_base_features"] = int(X.shape[1]); r["degree"] = degree
            out["runs"][f"{cfg}__{prot}"] = r
            rep = r["report"]
            print(f"{cfg:22s} {prot:20s} acc {r['accuracy_pooled']:.4f} (fold {r['accuracy_fold_mean']:.4f} ± {r['accuracy_fold_sd']:.4f})"
                  f" | dementia P {rep['1']['precision']:.2f} R {rep['1']['recall']:.2f} | healthy P {rep['0']['precision']:.2f} R {rep['0']['recall']:.2f}"
                  f" | CM {r['confusion_matrix']}")
    out["accuracy_pooled"] = out["runs"]["per_channel_deg2__participant_grouped"]["accuracy_pooled"]
    save(out, "lead_cohort176", a.outdir)
